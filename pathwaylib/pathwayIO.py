"""
pathwayIO: Read/write CSV for Pathway objects.

Write-out format merges edges with the same (source, target) pair:
  CH2,HCO,"(3,57)","(O,O2)",0.000229783+0.000713741

Input CSV format (from OpenFOAM):
  source,target,reaction_label,with,flux,hrr
"""

import csv
import cantera as ct
from .pathway import Pathway
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from .stiffsolver import stiff_solver


def _stiff_solver_wrapper(args):
    """Top-level wrapper for multiprocessing (lambdas can't be pickled)."""
    return stiff_solver(*args)

def read_csv(filepath: str) -> Pathway:
    """Read a CSV file and return a Pathway object."""
    p = Pathway()
    p.load_csv(filepath)
    return p


# def write_csv(pathway: Pathway, filepath: str):
#     """
#     Write a Pathway to CSV, merging edges with the same (source, target) pair.

#     For merged edges:
#       reaction_label becomes a parenthesized comma-separated list
#       with becomes a parenthesized comma-separated list
#       flux becomes a '+'-separated sum string
#     """
#     # Group edges by (source, target)
#     grouped: dict[tuple, list] = {}
#     edges = pathway.get_edges()
#     for source, target, rl, w, flux in edges:
#         key = (source, target)
#         if key not in grouped:
#             grouped[key] = []
#         grouped[key].append({"reaction_label": rl, "with": w, "flux": flux})

#     with open(filepath, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["source", "target", "reaction_label", "with", "flux"])

#         for (source, target), edge_list in sorted(grouped.items()):
#             if len(edge_list) == 1:
#                 e = edge_list[0]
#                 rl = _format_label(e["reaction_label"])
#                 w = _format_label(e["with"])
#                 writer.writerow([source, target, rl, w, e["flux"]])
#             else:
#                 # Merge: collect all reaction_labels, withs, fluxes
#                 rls = []
#                 ws = []
#                 fluxes = []
#                 for e in edge_list:
#                     rls.append(_format_label(e["reaction_label"]))
#                     ws.append(_format_label(e["with"]))
#                     fluxes.append(e["flux"])

#                 merged_rl = "(" + ",".join(rls) + ")"
#                 merged_w = "(" + ",".join(ws) + ")"
#                 merged_flux = "+".join(str(f) for f in fluxes)
#                 writer.writerow([source, target, merged_rl, merged_w, merged_flux])


# def _format_label(label):
#     """Format a reaction_label or with value for output."""
#     if isinstance(label, tuple):
#         return "(" + ",".join(str(x) for x in label) + ")"
#     return str(label)



# ---------------------------------------------------------------------------
# Reaction information (Etable)
# ---------------------------------------------------------------------------

def reaction_info(mechanism, species_to_trace="Al"):
    """Build Etable: atom-transfer edges for *species_to_trace* across all reactions.

    Each Etable entry is a dict::

        {
            "source": str,    # reactant species containing the traced element
            "target": str,    # product  species containing the traced element
            "coeffs": float,  # fractional weight of this source→target transfer
            "labels": int,    # 0-based reaction index
            "rwith" : list,   # other reactant species (co-reactants)
            "pwith" : list,   # other product  species (co-products)
        }
    """
    gas = ct.Solution(mechanism, verbose=0)
    n_reactions = gas.n_reactions

    # Count traced-element atoms on the reactant side per reaction
    total_species_atoms = np.zeros(n_reactions)
    for i, rxn in enumerate(gas.reactions()):
        atoms_set = set()
        for sp in rxn.reactants:
            atoms_set.update(gas.species(sp).composition.keys())
        if species_to_trace in atoms_set:
            for sp, coeff in rxn.reactants.items():
                total_species_atoms[i] += coeff * gas.species(sp).composition.get(species_to_trace, 0)

    Etable = []
    for i, rxn in enumerate(gas.reactions()):
        # Skip reactions that don't involve the traced element on the reactant side
        if total_species_atoms[i] == 0:
            continue
        all_reactants = set(rxn.reactants.keys())
        all_products  = set(rxn.products.keys())
        for sp_r, coeff_r in rxn.reactants.items():
            # Skip co-reactants that don't carry the traced element (e.g. O2 when tracing Al)
            if not gas.species(sp_r).composition.get(species_to_trace, 0):
                continue
            for sp_p, coeff_p in rxn.products.items():
                # Skip co-products that don't carry the traced element (e.g. O when tracing Al)
                if not gas.species(sp_p).composition.get(species_to_trace, 0):
                    continue
                sp_r_N = gas.species(sp_r).composition[species_to_trace]
                sp_p_N = gas.species(sp_p).composition[species_to_trace]
                # coeffs: fractional contribution of this sp_r→sp_p transfer,
                # weighted by stoichiometry and normalised by total traced-element
                # atoms on the reactant side of this reaction
                Etable.append({
                    "source": sp_r,
                    "target": sp_p,
                    "coeffs": float(sp_r_N * coeff_r) * float(sp_p_N * coeff_p)
                               / float(total_species_atoms[i]),
                    "labels": i,
                    "rwith":  list(all_reactants - {sp_r}),
                    "pwith":  list(all_products  - {sp_p}),
                })
    return Etable


# ---------------------------------------------------------------------------
# Pathway flux CSV
# ---------------------------------------------------------------------------

def write_csv(mechanism, species_to_trace, dataframe, filter_dict, deltaT, outputfile):
    """Integrate pathway fluxes over x ∈ [lrange, yrange] and write CSV.

    Output CSV columns: source, target, reaction_label, with, flux, hrr

    Returns
    -------
    hrr_data : dict[str, float]
        Maps ``"sourcetotarget"`` → net accumulated HRR (W/m³, summed over
        spatial points).  Positive = heat release; negative = heat absorption.
        Pass this to :func:`python.visualize.createDiag` as ``hrr_data``
        when using ``edge_mode='hrr'``.
    """
    Etable = reaction_info(mechanism, species_to_trace)
    
    # filter t
    # df_f   = dataframe[(dataframe["x"] >= lrange) & (dataframe["x"] <= yrange)].reset_index(drop=True)
    df_f = prepare_pathway_dataframe(mechanism, dataframe, deltaT, filter_dict)
    
    entry_strength = np.zeros(len(Etable))
    entry_hrr      = np.zeros(len(Etable))

    for i in range(len(df_f)):
        for e_idx, e in enumerate(Etable):
            r  = e["labels"]
            rr = df_f.loc[i, f"progress_R{r + 1}"]
            entry_strength[e_idx] += rr * e["coeffs"]
            entry_hrr[e_idx]      += df_f.loc[i, f"hrr_R{r + 1}"]

    hrr_data = {}  # "sourcetotarget" → accumulated HRR

    with open(outputfile, "w") as f:
        f.write("source,target,reaction_label,with,flux,hrr\n")

    for i, es in enumerate(entry_strength):
        if np.abs(es) <= 1e-18:
            continue
        e = Etable[i]
        if es > 0:
            source, target, with_sp = e["source"], e["target"], e["rwith"]
        else:
            source, target, with_sp = e["target"], e["source"], e["pwith"]
            es = -es

        key = source + "to" + target
        hrr_data[key] = hrr_data.get(key, 0.0) + entry_hrr[i]

        with open(outputfile, "a") as f:
            f.write(f"{source},{target},R{e['labels'] + 1},{'|'.join(with_sp)},{es},{entry_hrr[i]}\n")

    return hrr_data


# For given dataframe with YTP, check it and filter it, according to filter_list
def prepare_pathway_dataframe(mechanism, dataframe, deltaT, filter_dict:dict):
    gas = ct.Solution(mechanism, verbose=0)
    species_names = gas.species_names
    df_species_names = set(dataframe.columns)
    for sp in species_names:
        if sp not in df_species_names:
            raise ValueError(f"Species '{sp}' from the mechanism is missing in the dataframe columns.")
    if ("T" not in df_species_names) or ("p" not in df_species_names):
        raise ValueError("Dataframe must contain 'T' and 'p' columns.")
    
    # Filter it
    for filter_key, filter_value in filter_dict.items():
        condition = dataframe[filter_key].between(filter_value[0], filter_value[1])
        dataframe = dataframe[condition].reset_index(drop=True)
    
    # create new dataframe for per-reaction reaction rates and HRR
    prr_list = ["progress_R" + str(i+1) for i in range(gas.n_reactions)]
    hrr_list = ["hrr_R" + str(i+1) for i in range(gas.n_reactions)]
    df_R = pd.DataFrame(columns=prr_list + hrr_list)
    
    # undergo stiff solver (parallel)
    args_list = []
    for i in range(len(dataframe)):
        Y = {sp: dataframe.loc[i, sp] for sp in species_names}
        T = dataframe.loc[i, "T"]
        p = dataframe.loc[i, "p"]
        args_list.append((Y, T, p, deltaT, mechanism))

    nprocs = cpu_count()
    print(f"Running stiff solver on {len(args_list)} points using {nprocs} cores...")
    with Pool(processes=nprocs) as pool:
        results = list(tqdm(pool.imap(_stiff_solver_wrapper, args_list),
                            total=len(args_list), desc="Stiff solver"))

    for i, (_, _, _, _, xi, Q_dot_rxn, _, _) in enumerate(results):
        df_R.loc[i, prr_list] = xi
        df_R.loc[i, hrr_list] = Q_dot_rxn
        
    # concatenate the original and new dataframes
    df_final = pd.concat([dataframe, df_R], axis=1)
    return df_final

    

    