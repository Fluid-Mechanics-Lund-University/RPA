import numpy as np
from .pathway import Pathway
from copy import deepcopy
from graphviz import Digraph


def pathway_to_template(p: Pathway, edgeThres: float, entryThres: float) -> dict:
    """将 Pathway 对象转换为可视化用的 template 字典。

    内部通过 p.get_edges_with_dual_flux() 获取 edges 列表，每个元素结构::

        {
            "source": "Al",            # 源物种
            "target": "AlO",           # 目标物种
            "reaction_label": "R9",    # 反应编号
            "with": "O2",             # 共反应物（|分隔）
            "flux_down": 0.5,          # downstream 方向的通量
            "flux_up": 0.5,            # upstream 方向的通量
        }

    输出 template 结构（按 "源to目标" 聚合同一对物种间的多条反应）::

        {
            "AltoAlO": {                        # key: "源物种to目标物种"
                "flux": 0.85,                   # 该边所有反应的总通量
                "entries": [                    # 贡献该边的各条反应
                    {"with": "O2", "flux": 0.5, "label": "R9"},
                    {"with": "O",  "flux": 0.35, "label": "R1"},
                ]
            },
        }

    通量取 downstream 和 upstream 的均值，以平衡两个方向的差异。
    经过 entryThres 裁剪后，每条边只保留覆盖 (1 - entryThres) 通量的主要反应。
    经过 edgeThres 裁剪后，删除通量低于 max_flux * edgeThres 的弱边。
    """
    if not isinstance(p, Pathway):
        raise ValueError("Input must be an instance of the Pathway class.")
    template = {}
    edges = p.get_edges_with_dual_flux()
    for edge in edges:
        key  = edge["source"] + "to" + edge["target"]
        # 取上下行通量均值作为该边的代表通量
        flux = (edge["flux_up"] + edge["flux_down"]) / 2
        hrr  = edge.get("hrr", 0.0)
        if key in template:
            template[key]["flux"] += flux
            template[key]["hrr"]  += hrr
            template[key]["entries"].append({"with": edge["with"], "flux": flux, "label": edge["reaction_label"], "hrr": hrr})
        else:
            template[key] = {
                "flux":    flux,
                "hrr":     hrr,
                "entries": [{"with": edge["with"], "flux": flux, "label": edge["reaction_label"], "hrr": hrr}],
            }

    # 对每条边，只保留覆盖 (1 - entryThres) 通量的 entry，丢弃尾部次要反应
    for key in template:
        entries = template[key]["entries"]
        total   = sum(e["flux"] for e in entries)
        entries.sort(key=lambda e: e["flux"], reverse=True)
        cum = np.cumsum([e["flux"] / total for e in entries])
        cut = np.where(cum > 1 - entryThres)[0]
        keep = cut[0] + 1 if len(cut) > 0 else len(entries)
        template[key]["entries"] = entries[:keep]

    template = _remove_trivial_edges(template, edgeThres)
    return template


def _remove_trivial_edges(template: dict, threshold: float) -> dict:
    """删除通量低于 threshold * max_flux 的弱边。"""
    max_flux = max(template[k]["flux"] for k in template)
    cutoff   = threshold * max_flux
    return {k: v for k, v in template.items() if v["flux"] >= cutoff}


def combine_templates(template1: dict, template2: dict) -> dict:
    """合并两个 template，相同边的通量相加。"""
    template1 = deepcopy(template1)
    for key in template2:
        if key in template1:
            template1[key]["flux"] += template2[key]["flux"]
            template1[key]["hrr"]  = template1[key].get("hrr", 0.0) + template2[key].get("hrr", 0.0)
        else:
            template1[key] = template2[key]
    return template1


def combine_pathway_list(pathway_list: list, edgeThres: float, entryThres: float) -> dict:
    """将多个 Pathway 对象合并为一个 template，用于多条件叠加对比。"""
    if not all(isinstance(p, Pathway) for p in pathway_list):
        raise ValueError("All inputs must be Pathway instances.")
    if len(pathway_list) < 2:
        raise ValueError("At least two pathways are required.")

    combined = combine_templates(
        pathway_to_template(pathway_list[0], edgeThres, entryThres),
        pathway_to_template(pathway_list[1], edgeThres, entryThres),
    )
    for p in pathway_list[2:]:
        combined = combine_templates(combined, pathway_to_template(p, edgeThres, entryThres))
    return combined


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def createDiagramData(
    object: Pathway,
    refList: list = None,
    edgeThres: float = 0.1,
    entryThres: float = 0.1,
    neighborThres: float = 0.1,
    aroundSpecies: str = None,
    aroundN: int = None,
):
    """准备 linkDict 及物种/边集合，供 createDiag 渲染。

    返回的 linkDict 结构::

        {
            "AltoAlO": {
                "label": "Total flux:8.50e-01\\nR9(O2) 5.00e-01\\nR1(O) 3.50e-01",
                                                # 用于 flux 模式的边标签文本
                "strength": 0.85,               # 该边总通量（用于边宽缩放）
                "visible": True,                # 当前对象是否有此边
                "entries": [                    # 各反应的明细（combined 模式用）
                    {"with": "O2", "flux": 0.5, "label": "R9"},
                    {"with": "O",  "flux": 0.35, "label": "R1"},
                ],
            },
            "AlOAltoAl": {                      # 仅存在于 refList 的边：占位用
                "label": 0.12,                  # 此时 label 存的是 ref 的 flux 数值
                "strength": 0.0,
                "visible": False,               # 当前对象无此边，渲染为 invis
            },
        }

    Parameters
    ----------
    object        : Pathway to visualise
    refList       : reference Pathway list used to fix the layout skeleton
    edgeThres     : hide edges whose flux < edgeThres * max_flux
    entryThres    : keep only entries covering (1-entryThres) of each edge flux
    neighborThres : passed to keep_N_around
    aroundSpecies : if set, trim graph to N hops around this species
    aroundN       : number of hops for the around filter
    """
    object  = deepcopy(object)
    refList = deepcopy(refList) if refList is not None else None

    if refList is None:
        refList = [object]
    elif object not in refList:
        refList.append(object)

    if aroundSpecies is not None and aroundN is not None:
        for p in refList:
            p.keep_N_around(aroundSpecies, aroundN, neighborThres)

    templateObj = pathway_to_template(object, edgeThres, entryThres)
    templateRef = (
        pathway_to_template(refList[0], edgeThres, entryThres)
        if len(refList) == 1
        else combine_pathway_list(refList, edgeThres, entryThres)
    )

    # 用参考 template 构建 linkDict 骨架，保证布局稳定（即使当前对象没有该边也占位）
    linkDict = {
        key: {"label": templateRef[key]["flux"], "strength": 0.0, "visible": False, "hrr": 0.0}
        for key in templateRef
    }

    # 将当前对象的数据叠加到 linkDict 上
    for key, tval in templateObj.items():
        total_flux = tval["flux"]
        linkDict.setdefault(key, {"label": "", "strength": 0.0, "visible": False, "entries": [], "hrr": 0.0})
        linkDict[key]["label"]    = ""
        linkDict[key]["strength"] = total_flux
        linkDict[key]["visible"]  = True
        linkDict[key]["entries"]  = tval["entries"]
        linkDict[key]["hrr"]      = tval.get("hrr", 0.0)

    # 推导出参考集和当前对象的物种集合与边集合
    speciesRef, speciesObj = set(), set()
    edgesRef,   edgesObj   = [], []
    for key, val in linkDict.items():
        src, tgt = key.split("to", 1)
        speciesRef.add(src); speciesRef.add(tgt)
        edgesRef.append((src, tgt))
        if val["visible"]:
            speciesObj.add(src); speciesObj.add(tgt)
            edgesObj.append((src, tgt))

    return linkDict, speciesObj, speciesRef, edgesObj, edgesRef


# ---------------------------------------------------------------------------
# Diagram rendering
# ---------------------------------------------------------------------------

def _hrr_color(hrr_val: float, max_hrr: float) -> str:
    """根据 HRR 值返回 RGBA 十六进制颜色字符串。

    |hrr| < 50% max_hrr 时显示黑色（中性）；
    超过阈值后，放热为红色（#CC0000），吸热为蓝色（#0055CC），
    透明度 alpha 在 [0.5, 1.0] 区间内从 102 线性增至 255。
    """
    ratio = abs(hrr_val) / max_hrr if max_hrr > 0 else 0.0
    cutoff = 0.1
    if ratio < cutoff:
        return "#000000FF"   # black
    alpha = int(102 + (255 - 102) * (ratio - cutoff) / (1 - cutoff))   # 102..255 over [cutoff, 1.0]
    alpha = min(255, alpha)
    if hrr_val >= 0:
        return f"#CC0000{alpha:02X}"
    else:
        return f"#0055CC{alpha:02X}"


def createDiag(
    object: Pathway,
    refList: list = None,
    edgeThres: float = 0.1,
    entryThres: float = 0.1,
    neighborThres: float = 0.1,
    aroundSpecies: str = None,
    aroundN: int = None,
    highlightSpecies: list = None,
    figsize: tuple = None,
    dpi: int = 96,
):
    """Render a reaction pathway diagram.

    Parameters
    ----------
    object           : Pathway to visualise
    refList          : reference Pathway list for layout skeleton
    edgeThres        : relative flux threshold for showing an edge
    entryThres       : entry coverage threshold (see pathway_to_template)
    neighborThres    : threshold for keep_N_around
    aroundSpecies    : trim to N hops around this species
    aroundN          : number of hops
    highlightSpecies : species to highlight with a yellow fill
    figsize          : (width, height) in inches — controls the layout canvas size
    dpi              : output resolution in dots per inch (default 96).
                       Increase for higher-resolution PNG exports.
    Per-reaction HRR is read from the Pathway object (loaded from CSV with ``hrr``
    column).  Edge color encodes HRR (red = heat release, blue = heat absorption)
    with alpha ∝ |HRR| magnitude.  Each reaction entry shows its own HRR
    percentage.

    Returns
    -------
    graphviz.Digraph
    """
    linkDict, speciesObj, speciesRef, edgesObj, edgesRef = createDiagramData(
        object, refList, edgeThres, entryThres, neighborThres, aroundSpecies, aroundN
    )


    # 计算归一化基准（用于边宽度和颜色强度的缩放）
    MaxStrength = max(linkDict[k]["strength"] for k in linkDict) or 1.0

    abs_hrr_values = [abs(linkDict[k].get("hrr", 0.0)) for k in linkDict]
    MaxHRR = max(abs_hrr_values) if any(v > 0 for v in abs_hrr_values) else 1.0

    # realThreshold = edgeThres * MaxStrength

    # 收集需要显示的物种（至少有一条可见边连接）
    plot_species = set()
    for src, tgt in edgesObj:
        key = src + "to" + tgt

        plot_species.add(src)
        plot_species.add(tgt)

    # 构建 Graphviz 有向图
    dot = Digraph()
    dot.attr(rankdir="TD", fontname="Arial")
    dot.attr("node", shape="box", style="rounded", fontname="Arial")
    dot.attr("edge", fontname="Arial")
    dot.attr(dpi=str(dpi))
    if figsize is not None:
        dot.attr(size=f"{figsize[0]},{figsize[1]}", ratio="compress")

    highlight_set = set(highlightSpecies) if highlightSpecies else set()

    for s in speciesRef:
        if s in plot_species:
            if s in highlight_set:
                # 高亮物种：黄色填充
                dot.node(s, s, style="rounded,filled", fillcolor="#FFCC00", penwidth="2")
            else:
                dot.node(s, s)
        else:
            # 参考骨架中存在但当前不显示的物种：隐形占位，保持布局稳定
            dot.node(s, "", style="invis", width="0", height="0")

    for src, tgt in edgesRef:
        key  = src + "to" + tgt
        info = linkDict[key]

        if (src, tgt) not in edgesObj or not info["visible"]:
            dot.edge(src, tgt, style="invis")
            continue

        # 边宽由通量决定，颜色由 HRR 决定（透明度反映强度）
        edge_val = info["strength"]

        penwidth   = f"{max(0.5, 5.0 * edge_val / MaxStrength):.2f}" ## ensure minimum 0.5, maximum 5
        hrr_val    = info.get("hrr", 0.0)
        color      = _hrr_color(hrr_val, MaxHRR)
        total_flux = info["strength"] or 1.0
        entries    = info.get("entries", [])

        # 逐反应生成标签行，每条反应显示各自的 HRR 百分比（保留正负号）
        label_lines = ["HRR/Flux:"]
        shown_flux = 0.0
        for entry in entries:
            rxn     = str(entry["label"])
            with_sp = str(entry["with"]).strip() if entry["with"] else ""
            flux_pct = entry["flux"] / MaxStrength * 100 if MaxStrength > 0 else 0.0
            entry_hrr = entry.get("hrr", 0.0)
            entry_hrr_pct = entry_hrr / MaxHRR * 100 if MaxHRR > 0 else 0.0
            shown_flux += entry["flux"]
            if with_sp:
                with_str = with_sp.replace("|", ",")
                rxn_line = f"{rxn}: (+{with_str})"
            else:
                rxn_line = f"{rxn}: decomp"
            label_lines.append(rxn_line)
            label_lines.append(f"{entry_hrr_pct:+.1f}J / {flux_pct:.1f}mol")

        # 若 entryThres 过滤掉了部分反应，显示剩余百分比提示
        hidden_pct = (total_flux - shown_flux) / MaxStrength * 100 if MaxStrength > 0 else 0.0
        if hidden_pct > 0.5:   # 忽略舍入噪声
            label_lines.append(f"(+other {hidden_pct:.0f} mol)")
        edge_label = "\n".join(label_lines)
        dot.edge(src, tgt, label=edge_label, penwidth=penwidth, color=color,
                    fontcolor=color)
        
    print("Maximum HRR:{:2e} W/m³".format(MaxHRR))
    print("Maximum flux:{:2e} kmol/m³/s".format(MaxStrength))



    return dot
