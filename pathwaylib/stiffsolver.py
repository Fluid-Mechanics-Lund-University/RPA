"""
metal.py — Core analysis functions for Al combustion sampling-line post-processing.

Functions
---------
stiff_solver          : integrate gas-phase kinetics at one spatial point
reaction_info         : build Etable of traced-element atom-transfer edges
write_out_pathway     : integrate pathway fluxes → CSV; returns HRR dict per edge

Typical Usage
-------------
Step 1 — Generate pathway CSV and HRR data from a simulation sampling line,
    note that the simulation sample line should contain Y,T,p and rho:

    import pandas as pd
    from metal import write_out_pathway, write_reaction_hrr_csv

    df = pd.read_csv("data/line_with_reactions.csv")

    hrr_data = write_out_pathway(
        mechanism      = "./Glorian.yaml",
        species_to_trace = "Al",
        dataframe      = df,
        lrange         = 0.001,   # x-range start (m)
        yrange         = 0.005,   # x-range end   (m)
        outputfile     = "csvs/Al_pathway.csv",
    )


Step 2 — Load the pathway CSV and visualize:

Note: stiff_solver is called internally by write_out_pathway for each spatial
point; call it directly only if you need per-point kinetics output.
"""

import numpy as np
import cantera as ct

MECHANISM = "./Glorian.yaml"
PRESSURE  = 101325  # Pa


# ---------------------------------------------------------------------------
# Stiff ODE solver
# ---------------------------------------------------------------------------

def stiff_solver(Y, T, p, deltaT, mechanism=MECHANISM):
    """Integrate gas-phase kinetics for *deltaT* seconds using a stiff solver.

    Parameters
    ----------
    Y        : dict or array  – mass fractions
    T        : float          – temperature (K)
    p        : float          – pressure (Pa)
    deltaT   : float          – integration time (s)
    mechanism: str            – path to Cantera YAML mechanism

    Returns
    -------
    Y_end, T_end, p_end, Q_dot, xi, Q_dot_rxn
        Y_end     : mass fractions after integration
        T_end     : temperature after integration (K)
        p_end     : pressure after integration (Pa)
        Q_dot     : time-averaged total heat release rate (W/m³)
        xi        : per-reaction progress integrated over deltaT (kmol/m³)
        Q_dot_rxn : time-averaged per-reaction heat release rate (W/m³)
        dY/dt     : reaction rate in mass fraction units (1/s)
        dX/dt     : reaction rate in mole fraction units (1/s)
    """
    gas = ct.Solution(mechanism, verbose=0)
    gas.TPY = T, p, Y

    reactor = ct.IdealGasReactor(gas)
    sim     = ct.ReactorNet([reactor])
    sim.rtol = 1e-3
    sim.atol = 1e-11
    sim.max_time_step  =deltaT
    
    Yorg = gas.Y.copy()  # for later dY/dt calculation

    xi     = np.zeros(gas.n_reactions)
    Q_rxn  = np.zeros(gas.n_reactions)
    t      = 0.0
    q_prev = gas.net_rates_of_progress.copy()   # initialise at t=0, not zeros
    h_prev = gas.heat_production_rates.copy()
    
    t_before = 0.0
    dt = 0.0 

    while t < deltaT:
        t_before = t
        if (t+dt) > deltaT:
            sim.advance(deltaT)
            t = deltaT
        else:
            t = sim.step()
        dt = t - t_before

        q_curr = gas.net_rates_of_progress.copy()
        h_curr = gas.heat_production_rates.copy()

        xi    += 0.5 * (q_prev + q_curr) * dt
        Q_rxn += 0.5 * (h_prev + h_curr) * dt

        q_prev, h_prev = q_curr, h_curr
    
    dY = gas.Y - Yorg
    dX = dY * gas.mean_molecular_weight / gas.molecular_weights

    return gas.Y.copy(), gas.T, gas.P, float(np.sum(Q_rxn)) / deltaT, xi/deltaT, Q_rxn / deltaT, dY/deltaT, dX/deltaT




