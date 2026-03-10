"""
Pathway class: stores upstream/downstream connections for each species,
"""

import csv
import numpy as np
from copy import deepcopy


class Pathway:
    """双向有向图，存储各物种之间的反应通量连接关系。

    每条边存储两次——在源物种的 downstream 和目标物种的 upstream 各存一份，
    使得两个方向的通量可以独立查询。这种双向存储在元素追踪中非常重要：
    消除中间物种后，下行通量（X 有多少流向 Y）和上行通量（Y 有多少来自 X）可能不同。

    数据结构::

        self.species[name] = {
            "downstream": {
                neighbor_a: [{"reaction_label": ..., "with": ..., "flux": float}, ...],
            },
            "upstream": {
                neighbor_b: [...],
            }
        }
    """

    def __init__(self):
        # edge dict 结构: {"reaction_label": str|tuple, "with": str|tuple, "flux": float}
        # 经过消除操作后，reaction_label 和 with 会变成嵌套 tuple，例如 ("R1", "R2")
        # elimination has been removed
        self.species: dict[str, dict] = {}

    def _ensure_species(self, name: str):
        if name not in self.species:
            self.species[name] = {"downstream": {}, "upstream": {}}

    def add_edge(self, source: str, target: str, reaction_label, with_species, flux: float, hrr: float = 0.0):
        """添加一条从 source 到 target 的有向边。"""
        self._ensure_species(source)
        self._ensure_species(target)

        edge = {"reaction_label": reaction_label, "with": with_species, "flux": flux, "hrr": hrr}

        # 加入 source 的下行邻居列表
        if target not in self.species[source]["downstream"]:
            self.species[source]["downstream"][target] = []
        self.species[source]["downstream"][target].append(edge)

        # 加入 target 的上行邻居列表
        if source not in self.species[target]["upstream"]:
            self.species[target]["upstream"][source] = []
        self.species[target]["upstream"][source].append(
            {"reaction_label": reaction_label, "with": with_species, "flux": flux, "hrr": hrr}
        )

    def load_csv(self, filepath: str):
        """读取 CSV 文件，构建双向嵌套字典。"""
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = row["source"].strip()
                target = row["target"].strip()
                reaction_label = row["reaction_label"].strip()
                with_species = row["with"].strip() if row["with"] else ""
                flux = float(row["flux"])
                hrr = float(row["hrr"]) if "hrr" in row and row["hrr"] else 0.0
                self.add_edge(source, target, reaction_label, with_species, flux, hrr)

    def downstream_total(self, species: str) -> float:
        """计算某物种所有下行通量之和。"""
        total = 0.0
        for neighbor, edges in self.species[species]["downstream"].items():
            for e in edges:
                total += e["flux"]
        return total

    def upstream_total(self, species: str) -> float:
        """计算某物种所有上行通量之和。"""
        total = 0.0
        for neighbor, edges in self.species[species]["upstream"].items():
            for e in edges:
                total += e["flux"]
        return total
    

    def keep_N_around(self, who: str, N: int, neighborThres: float = 0.1):
        """保留以 who 为中心、N 跳以内的子图，按累积通量贡献裁剪邻居。

        neighborThres=0.1 表示只保留覆盖 90% 通量的邻居，丢弃尾部弱边。
        """
        # 记录需要保留的边（格式："source->target"）
        relevant_pairs = set()
        upper_relevant_species = set()  # 下行方向遍历到的物种
        lower_relevant_species = set()  # 上行方向遍历到的物种

        # --- 向下游遍历 N 跳 ---
        fromList = [who]
        for level in range(N):
            next_front = set()
            for sp in fromList:
                strength_list = []
                neighbor_list = []
                for d_neighbor in self.species[sp]["downstream"]:
                    strength = sum(e["flux"] for e in self.species[sp]["downstream"][d_neighbor])
                    strength_list.append(strength)
                    neighbor_list.append(d_neighbor)
                if not strength_list:
                    continue
                # 按通量从大到小排序，计算累积贡献，截断尾部
                index = np.argsort(strength_list)[::-1]
                strength_list = [strength_list[i] for i in index]
                neighbor_list = [neighbor_list[i] for i in index]
                total = sum(strength_list)
                cum_contribution = np.cumsum(strength_list) / total
                cut_index = np.where(cum_contribution > 1-neighborThres)[0]
                keep_count = cut_index[0] + 1 if len(cut_index) > 0 else len(strength_list)

                for i in range(keep_count):
                    relevant_pairs.add(sp + "->" + neighbor_list[i])
                    upper_relevant_species.add(neighbor_list[i])
                    next_front.add(neighbor_list[i])
            fromList = list(next_front)
            
        # --- 向上游遍历 N 跳 ---
        fromList = [who]
        for level in range(N):
            next_front = set()
            for sp in fromList:
                strength_list = []
                neighbor_list = []
                for u_neighbor in self.species[sp]["upstream"]:
                    strength = sum(e["flux"] for e in self.species[sp]["upstream"][u_neighbor])
                    strength_list.append(strength)
                    neighbor_list.append(u_neighbor)
                if not strength_list:
                    continue
                index = np.argsort(strength_list)[::-1]
                strength_list = [strength_list[i] for i in index]
                neighbor_list = [neighbor_list[i] for i in index]
                total = sum(strength_list)
                cum_contribution = np.cumsum(strength_list) / total
                cut_index = np.where(cum_contribution > 1-neighborThres)[0]
                keep_count = cut_index[0] + 1 if len(cut_index) > 0 else len(strength_list)

                for i in range(keep_count):
                    relevant_pairs.add(neighbor_list[i] + "->" + sp)
                    lower_relevant_species.add(neighbor_list[i])
                    next_front.add(neighbor_list[i])
            fromList = list(next_front)

        # --- 清理：删除不在 relevant_pairs 中的边 ---
        for sp in self.species:
            down_stream_list = list(self.species[sp]["downstream"].keys())
            up_stream_list = list(self.species[sp]["upstream"].keys())
            for d_neighbor in down_stream_list:
                add_string = sp + "->" + d_neighbor
                if add_string not in relevant_pairs:
                    del self.species[sp]["downstream"][d_neighbor]
            for u_neighbor in up_stream_list:
                add_string = u_neighbor + "->" + sp
                if add_string not in relevant_pairs:
                    del self.species[sp]["upstream"][u_neighbor]
                    
        # clean irrelevant species
        for sp in list(self.species.keys()):
            if (len(self.species[sp]["downstream"]) == 0 and len(self.species[sp]["upstream"]) == 0):
                del self.species[sp]
        return None


    def get_edges(self) -> list[tuple]:
        """返回所有边，格式为 (source, target, reaction_label, with, flux)。
        仅从 downstream 视角遍历，避免重复。
        """
        edges = []
        for source, data in self.species.items():
            for target, edge_list in data["downstream"].items():
                for e in edge_list:
                    edges.append((
                        source, target,
                        e["reaction_label"], e["with"], e["flux"],
                        e.get("hrr", 0.0)
                    ))
        return edges

    def get_edges_with_dual_flux(self) -> list[dict]:
        """返回所有边，同时包含 downstream 和 upstream 两个方向的通量值。
        按 (source, target, reaction_label, with) 分组配对。
        消除操作后两个方向的值可能不同。
        """
        # 从 downstream 视角收集
        down_edges = {}
        for source, data in self.species.items():
            for target, edge_list in data["downstream"].items():
                for e in edge_list:
                    key = (source, target, str(e["reaction_label"]), str(e["with"]))
                    down_edges[key] = {"flux": e["flux"], "hrr": e.get("hrr", 0.0)}

        # 从 upstream 视角收集
        up_edges = {}
        for target, data in self.species.items():
            for source, edge_list in data["upstream"].items():
                for e in edge_list:
                    key = (source, target, str(e["reaction_label"]), str(e["with"]))
                    up_edges[key] = e["flux"]

        # 合并两个视角
        all_keys = set(down_edges.keys()) | set(up_edges.keys())
        result = []
        for key in sorted(all_keys):
            source, target, rl, w = key
            down_info = down_edges.get(key, {"flux": 0.0, "hrr": 0.0})
            result.append({
                "source": source,
                "target": target,
                "reaction_label": rl,
                "with": w,
                "flux_down": down_info["flux"],
                "flux_up": up_edges.get(key, 0.0),
                "hrr": down_info["hrr"],
            })
        return result

    def get_species_list(self) -> list[str]:
        """返回图中所有物种名称的列表。"""
        return list(self.species.keys())

    def __repr__(self):
        edges = self.get_edges()
        lines = [f"Pathway with {len(self.species)} species, {len(edges)} edges:"]
        for e in edges:
            lines.append(f"  {e[0]} -> {e[1]}  reaction={e[2]} with={e[3]} flux={e[4]:.6g} hrr={e[5]:.6g}")
        return "\n".join(lines)
