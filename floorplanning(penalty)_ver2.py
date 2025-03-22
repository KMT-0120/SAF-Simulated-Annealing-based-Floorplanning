# 해당 주석 문단은 지우지말것(특히 GPT 사용시, 지워지는지 또한 확인한 후 다시 추가), GPT 돌릴떄 함수 이름, 변수명 변경하지 말것
# 3/8 iter 5000번마다 plot, SA종료 후 T 변화 추이 그래프 plot 추가
# 사용하지 않는 과거 코드 삭제(adjust_for_overlaps, check_overlap, is_within_bounds)
# plot관련해서 노드 크기/폰트 크기 조정, 노드 간격 조정
# c factor 조정

# 3/9 GSRC 적용, 초기해 leftchain -> random으로 변경
# 3/15 왼쪽 자식의 배치가 부모노드와 너무 조금 떨어져있던 버그 수정
# 3/18 cost에 penalty추가, 정규화는 area_norm으로 해둠,
#     area_norm이 HPWL_norm과 너무 크게 차이나는것을 확인하였기에(EX:4, 2000) area_norm * 500해서 사용
#     초기, 마지막 비용값을 더 잘 확인하기 위해 get_nomalized_values()함수 추가(cal_cost와 매커니즘 같으니 cal_cost 수정시에 같이 수정해줄것)
#     default_parent를 1000, 1000대신 전체 module의 넓이*1.2로 설정
#     sample_moves가 버려지는 경우를 대비해 sample_moves 전부를 기억해두고 제일 좋았던 코스트 변화량을 적용
#     left-chain 적용

#3/21 penalty 함수 수정
#     > Area Violation Function(추가) + Excessive Length Violation Function
#     penalty_norm * 50 // 스케일 조정 
#     5000번 마다 plot 할 때, Area/HPWL/Penalty/Cost/Dead space 같이 표시, 최종 Penalty Evolution 플롯 추가 < ##(추가)로 표시, 사용안 할거면 주석처리

import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import random
import re
import sys

sys.setrecursionlimit(10000)


# ─────────────────────────────────────────────────────────────
# 1. Module, BTreeNode, Chip 클래스 (Random)
# ─────────────────────────────────────────────────────────────
class Module:
    def __init__(self, name: str, width: float, height: float, module_type: str, net=None):
        self.name = name
        self.width = width
        self.height = height
        self.area = width * height
        self.x = 0
        self.y = 0
        self.type = module_type
        self.net = net if net is not None else []
        self.order = None

    def rotate(self):
        """너비와 높이를 교환"""
        self.width, self.height = self.height, self.width

    def set_position(self, x: float, y: float, order: int):
        self.x = x
        self.y = y
        self.order = order

    def __str__(self):
        formatted_net = "\n    ".join(self.net)
        return (f"Module(Name={self.name}, "
                f"Width={self.width:.2f}, Height={self.height:.2f}, "
                f"Area={self.area:.2f}, Position=({self.x:.2f}, {self.y:.2f}), "
                f"Order={self.order}, Type={self.type}, Net=\n    {formatted_net})")


class BTreeNode:
    def __init__(self, module, parent=None):
        self.module = module
        self.left = None
        self.right = None
        self.parent = parent


class ContourNode:
    """Contour segment: [x1, x2) 구간에서 y2 높이"""

    def __init__(self, x1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y2 = y2


class Chip:
    def __init__(self, modules):
        """
        parent가 없는 경우에는 전체 module들의 area합 * 1.2와 동일한
        정사각형을 DefaultParent로 설정합니다.
        """
        # PARENT 모듈 외의 모든 모듈
        self.modules = [m for m in modules if m.type != 'PARENT']

        # PARENT 모듈이 있는지 확인
        self.bound = next((m for m in modules if m.type == 'PARENT'), None)
        if not self.bound:
            total_area = sum(m.area for m in self.modules)
            side = math.sqrt(total_area)
            self.bound = Module(
                name='DefaultParent',
                width=side,
                height=side,
                module_type='PARENT'
            )

        self.build_b_tree()

        self.contour_line = []
        self.max_width = 0
        self.max_height = 0

    def build_b_tree(self):
        """
        모든 모듈을 "왼쪽 체인"으로 연결한다.
        1) 첫 번째 모듈 -> root
        2) 두 번째 모듈 -> root.left
        3) 세 번째 모듈 -> root.left.left
        ... 식으로 연결
        """
        if not self.modules:
            self.root = None
            return

        # 첫 번째 모듈 -> 루트 노드
        self.root = BTreeNode(self.modules[0], parent=None)

        # 나머지 모듈을 차례로 왼쪽 체인으로 연결
        current = self.root
        for mod in self.modules[1:]:
            new_node = BTreeNode(mod, parent=current)
            current.left = new_node
            current = new_node

    def collect_all_nodes(self):
        if not self.root:
            return []
        result = []
        queue = [self.root]
        while queue:
            cur = queue.pop(0)
            result.append(cur)
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)
        return result

    # ─────────────────────────────────────────
    # B*-tree 연산들
    def rotate_node(self, node):
        node.module.rotate()

    def move_node(self, node):
        if node.parent is None:
            return f"[Op2: Move] Cannot move ROOT node ({node.module.name})."
        self.remove_node_from_parent(node)
        cands = self.find_possible_parents()
        if not cands:
            return "[Op2: Move] No available spot to re-insert node."
        new_par = random.choice(cands)
        side = self.attach_node(new_par, node)
        return f"[Op2: Move] Moved {node.module.name} under {new_par.module.name} ({side})"

    def remove_node_from_parent(self, node):
        p = node.parent
        if p is None:
            return
        if p.left == node:
            p.left = None
        elif p.right == node:
            p.right = None
        node.parent = None

    def find_possible_parents(self):
        all_nd = self.collect_all_nodes()
        results = []
        for nd in all_nd:
            if nd.left is None or nd.right is None:
                results.append(nd)
        return results

    def attach_node(self, parent, node):
        node.parent = parent
        slots = []
        if not parent.left:
            slots.append("left")
        if not parent.right:
            slots.append("right")
        if not slots:
            return "NoSlot"
        chosen = random.choice(slots)
        if chosen == "left":
            parent.left = node
        else:
            parent.right = node
        return chosen

    def swap_nodes(self, nodeA, nodeB):
        nodeA.module, nodeB.module = nodeB.module, nodeA.module

    def apply_random_operation(self):
        """
        랜덤 연산을 수행하고,
        해당 오퍼레이션/연산에 사용된 노드 정보를 함께 반환한다.

        반환 형식:
           (msg, op, op_data)
         여기서 op_data는 dict로, { 'op': ..., 'nodeA_name': ..., 'nodeB_name': ... } 형태
         ex) rotate -> {'op':'rotate', 'nodeA_name':...}
             move   -> {'op':'move',   'nodeA_name':...}
             swap   -> {'op':'swap',   'nodeA_name':..., 'nodeB_name':...}
        """
        ops = ["rotate", "move", "swap"]
        op = random.choice(ops)
        all_nodes = self.collect_all_nodes()
        if not all_nodes:
            print("[Warn] No nodes in the tree.")
            return ("NoNode", None, None)

        if op == "rotate":
            chosen = random.choice(all_nodes)
            self.rotate_node(chosen)
            msg = f"[Op1: Rotate] Rotated {chosen.module.name}"
            op_data = {
                'op': 'rotate',
                'nodeA_name': chosen.module.name
            }
        elif op == "move":
            chosen = random.choice(all_nodes)
            msg = self.move_node(chosen)
            op_data = {
                'op': 'move',
                'nodeA_name': chosen.module.name
            }
        else:  # swap
            if len(all_nodes) < 2:
                msg = "[Op3: Swap] Not enough nodes to swap."
                op_data = {'op': 'swap', 'nodeA_name': None, 'nodeB_name': None}
            else:
                nodeA, nodeB = random.sample(all_nodes, 2)
                self.swap_nodes(nodeA, nodeB)
                msg = f"[Op3: Swap] Swapped {nodeA.module.name} <-> {nodeB.module.name}"
                op_data = {
                    'op': 'swap',
                    'nodeA_name': nodeA.module.name,
                    'nodeB_name': nodeB.module.name
                }

        print(msg)
        # 좌표 초기화 & 재계산
        for nd in all_nodes:
            nd.module.x = 0
            nd.module.y = 0
            nd.module.order = None
        self.calculate_coordinates()
        return (msg, op, op_data)

    def reapply_operation(self, op_data):
        """
        sample 시 저장한 op_data대로 "동일한 노드/동일한 연산"을 다시 적용
        """
        if not op_data:
            return "[NoOpData]", "NoOp"
        all_nodes = self.collect_all_nodes()

        # 노드 이름 -> node 객체를 찾는다.
        def find_node_by_name(nm):
            for nd in all_nodes:
                if nd.module.name == nm:
                    return nd
            return None

        op_type = op_data.get('op', 'NoOp')
        if op_type == 'rotate':
            nodeA_name = op_data.get('nodeA_name')
            ndA = find_node_by_name(nodeA_name)
            if ndA is None:
                return "[Rotate] Node not found", op_type
            self.rotate_node(ndA)
            msg = f"[Op1: Rotate] Re-applied rotate on {nodeA_name}"
        elif op_type == 'move':
            nodeA_name = op_data.get('nodeA_name')
            ndA = find_node_by_name(nodeA_name)
            if ndA is None:
                return "[Move] Node not found", op_type
            msg = self.move_node(ndA)
            msg += " [Re-applied]"
        elif op_type == 'swap':
            nodeA_name = op_data.get('nodeA_name')
            nodeB_name = op_data.get('nodeB_name')
            ndA = find_node_by_name(nodeA_name)
            ndB = find_node_by_name(nodeB_name)
            if not ndA or not ndB:
                return "[Swap] Node(s) not found", op_type
            self.swap_nodes(ndA, ndB)
            msg = f"[Op3: Swap] Re-applied swap: {nodeA_name} <-> {nodeB_name}"
        else:
            msg = "[NoOp or invalid op]"
            op_type = "NoOp"

        # 재계산
        for nd in all_nodes:
            nd.module.x = 0
            nd.module.y = 0
            nd.module.order = None
        self.calculate_coordinates()

        return msg, op_type

    def randomize_b_tree(self, w=0.5):
        random.shuffle(self.modules)
        self.build_b_tree()
        self.calculate_coordinates()
        return calc_combined_cost(self.modules, w, chip=self)

    # ─────────────────────────────────────────
    # 좌표 배치 (Contour + DFS)
    def calculate_coordinates(self):
        if not self.root:
            return
        self.contour_line = []
        self.max_width = 0
        self.max_height = 0
        all_nd = self.collect_all_nodes()
        for nd in all_nd:
            nd.module.x = 0
            nd.module.y = 0
            nd.module.order = None
        self._dfs_place_node(self.root, 1, 0, 0)

    def _dfs_place_node(self, node, order, x_offset, y_offset):
        if not node:
            return order
        node.module.x = x_offset
        node.module.y = y_offset
        node.module.order = order

        x1 = x_offset
        x2 = x_offset + node.module.width
        top_y = y_offset + node.module.height

        self.insert_contour_segment(x1, x2, top_y)
        if x2 > self.max_width:
            self.max_width = x2
        if top_y > self.max_height:
            self.max_height = top_y
        order += 1

        # 왼쪽 => x+width
        if node.left:
            lx = node.module.x + node.module.width
            rx = lx + node.left.module.width
            ly = self.update_contour(lx, rx)
            order = self._dfs_place_node(node.left, order, lx, ly)

        # 오른쪽 => x
        if node.right:
            rxs = node.module.x
            rxe = rxs + node.right.module.width
            rys = self.update_contour(rxs, rxe)
            order = self._dfs_place_node(node.right, order, rxs, rys)

        return order

    def update_contour(self, x1, x2):
        base_y = 0
        for seg in self.contour_line:
            if not (seg.x2 <= x1 or seg.x1 >= x2):
                base_y = max(base_y, seg.y2)
        return base_y

    def insert_contour_segment(self, x1, x2, new_top):
        updated = []
        i = 0
        while i < len(self.contour_line):
            seg = self.contour_line[i]
            if seg.x2 <= x1 or seg.x1 >= x2:
                updated.append(seg)
            else:
                if seg.x1 < x1 and seg.x2 > x2:
                    updated.append(ContourNode(seg.x1, x1, seg.y2))
                    updated.append(ContourNode(x2, seg.x2, seg.y2))
                elif seg.x1 < x1 and seg.x2 <= x2:
                    updated.append(ContourNode(seg.x1, x1, seg.y2))
                elif seg.x1 >= x1 and seg.x2 > x2:
                    updated.append(ContourNode(x2, seg.x2, seg.y2))
            i += 1
        updated.append(ContourNode(x1, x2, new_top))
        updated.sort(key=lambda s: s.x1)
        self.contour_line = updated

    # plot( iteration=None ): 더 넓게 노드 배치 + 폰트/노드 크기 조정
    def plot_b_tree(self, iteration=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        # (1) 물리적 배치
        self._plot_node(self.root, ax1)
        bound_rect = plt.Rectangle((0, 0), self.bound.width, self.bound.height,
                                   edgecolor='red', facecolor='none', lw=2)
        ax1.add_patch(bound_rect)

        if iteration is not None:
            ax1.set_title(f"B*-Tree Physical Placement (Left-Chain, Iter={iteration})")
        else:
            ax1.set_title("B*-Tree Physical Placement (Left-Chain)")

        ax1.set_xlabel("X-coordinate")
        ax1.set_ylabel("Y-coordinate")
        ax1.set_xlim(0, self.bound.width + 200)
        ax1.set_ylim(0, self.bound.height + 200)

        # (2) B*-Tree 구조 (더 넓은 x간격, 작아진 노드/폰트)
        G = nx.DiGraph()
        pos = {}
        self.add_edges(self.root, None, 0, 0, G, pos)

        nx.draw(G, pos, ax=ax2, with_labels=True, node_color="lightblue",
                edge_color="black", node_size=600, font_size=7)
        if iteration is not None:
            ax2.set_title(f"B*-Tree Structure (Left-Chain, Iter={iteration})")
        else:
            ax2.set_title("B*-Tree Structure (Left-Chain)")

    def _plot_node(self, node, ax):
        if not node:
            return
        m = node.module
        face_c = 'lightblue'
        rect = plt.Rectangle((m.x, m.y), m.width, m.height,
                             edgecolor='blue', facecolor=face_c, fill=True, lw=2)
        ax.add_patch(rect)
        ax.text(m.x + m.width / 2, m.y + m.height / 2,
                f'{m.name}\n#{m.order}', ha='center', va='center', fontsize=8)
        self._plot_node(node.left, ax)
        self._plot_node(node.right, ax)

    def add_edges(self, node, parent_name, depth, x_pos, G, pos):
        """
        노드 간 거리를 벌리기 위해
        - 수평은 x_pos*2
        - 수직은 -depth*2
        오른쪽 자식 지나갈 때 x_pos+2
        """
        if not node:
            return x_pos
        node_name = node.module.name
        # (배율 확대)
        pos[node_name] = (x_pos * 2, -depth * 2)

        G.add_node(node_name, label=node.module.name)
        if parent_name:
            G.add_edge(parent_name, node_name)

        # 왼쪽
        x_pos = self.add_edges(node.left, node_name, depth + 1, x_pos, G, pos)
        # 오른쪽(간격 +2)
        x_pos = self.add_edges(node.right, node_name, depth + 1, x_pos + 2, G, pos)
        return x_pos


# ─────────────────────────────────────────────────────────────
# 2. Cost 계산 함수들
# ─────────────────────────────────────────────────────────────

def calculate_hpwl(modules):
    net_dict = {}
    for m in modules:
        for net_name in m.net:
            if net_name not in net_dict:
                net_dict[net_name] = []
            net_dict[net_name].append(m)
    if not net_dict:
        return 0
    total_hpwl = 0
    for net, mods in net_dict.items():
        min_x = min(mm.x for mm in mods)
        max_x = max(mm.x + mm.width for mm in mods)
        min_y = min(mm.y for mm in mods)
        max_y = max(mm.y + mm.height for mm in mods)
        net_hpwl = (max_x - min_x) + (max_y - min_y)
        total_hpwl += net_hpwl
    return total_hpwl


def calculate_total_area(modules):
    if not modules:
        return 0, 0, 0
    min_x = min(m.x for m in modules)
    max_x = max(m.x + m.width for m in modules)
    min_y = min(m.y for m in modules)
    max_y = max(m.y + m.height for m in modules)
    width = (max_x - min_x)
    height = (max_y - min_y)
    return width, height, (width * height)


def calc_combined_cost(modules, w=0.5, chip=None, r=1.0):
    """
    cost = w * area_norm + (1-w) * hpwl_norm + r * penalty_norm
    여기서 penalty = sum( (초과_width^2) + (초과_height^2) ) for all modules that cross boundary
    (penalty_norm = penalty / base_scale)
    """
    cost_scale = 100
    if not modules:
        return 0.0

    # (1) area, hpwl 스케일
    base_scale = sum(m.area for m in modules)
    net_area_sum = sum(m.area for m in modules if m.net)
    if net_area_sum == 0:
        net_area_sum = base_scale

    # (2) 현재 배치의 bounding box와 HPWL
    w_, h_, area_bb = calculate_total_area(modules)
    hpwl = calculate_hpwl(modules)

    # (3) penalty 계산 – EAL 방식 적용 (Area Violation Function + Excessive Length Violation)
    if chip:
        # Prescribed outline 크기를 가져옴 (PARENT 모듈 기준)
        bound_w = chip.bound.width  # 칩 경계의 너비
        bound_h = chip.bound.height  # 칩 경계의 높이

        # 전체 배치의 bounding box 크기를 계산합니다.
        # calculate_total_area() 함수는 (w_bb, h_bb, area_bb)를 반환합니다.
        w_bb, h_bb, area_bb = calculate_total_area(modules)

        # [Area Violation Function (A′)]
        # 이 함수는 전체 floorplan이 prescribed outline을 벗어난 면적을 계산합니다.
        # - 만약 floorplan이 outline 내에 완전히 있다면 A′ = 0.
        # - 너비만 초과하면: A′ = (w_bb - bound_w) * bound_h
        # - 높이만 초과하면: A′ = (h_bb - bound_h) * bound_w
        # - 둘 다 초과하면: A′ = (w_bb * h_bb) - (bound_w * bound_h)
        if w_bb <= bound_w and h_bb <= bound_h:
            area_violation = 0.0
        elif w_bb > bound_w and h_bb <= bound_h:
            area_violation = (w_bb - bound_w) * bound_h
        elif w_bb <= bound_w and h_bb > bound_h:
            area_violation = (h_bb - bound_h) * bound_w
        else:
            area_violation = (w_bb * h_bb) - (bound_w * bound_h)

        # [Excessive Length Violation Function (L′)]
        # 이 함수는 각 모듈이 prescribed outline을 벗어난 거리의 제곱합을 계산합니다.
        # 각 모듈에 대해, 오른쪽 및 상단 초과분을 구하고, 초과분이 있을 경우 제곱해서 누적합니다.
        length_violation = 0.0
        for mm in modules:
            over_w = (mm.x + mm.width) - bound_w  # 모듈의 오른쪽 초과 길이
            over_h = (mm.y + mm.height) - bound_h  # 모듈의 상단 초과 길이
            if over_w > 0:
                length_violation += over_w ** 2
            if over_h > 0:
                length_violation += over_h ** 2

        # 전체 penalty는 위 두 위반 항목의 합입니다.
        penalty_sum = area_violation + length_violation
    else:
        penalty_sum = 0.0

    # (4) 정규화
    area_norm = area_bb / base_scale if base_scale else 0
    hpwl_norm = hpwl / (2 * math.sqrt(net_area_sum)) if net_area_sum > 0 else 0
    penalty_norm = penalty_sum / base_scale if base_scale else 0


    # area_norm과 HPWL_norm의 스케일 차를 줄이기 위해 area_norm에 500 곱함
    area_norm *= 500
    # penalty_norm scale 조절
    penalty_norm *= 50

    # (5) 최종 cost
    cost_value = w * area_norm + (1 - w) * hpwl_norm + r * penalty_norm
    return cost_value * cost_scale


def get_normalized_values(modules, chip=None, w=0.5, r=1.0):
    """
    area_norm, hpwl_norm, penalty_norm을 추출하여 함께 반환.
    calc_combined_cost와 동일한 계산 로직을 쓴다.
    """
    if not modules:
        return (0.0, 0.0, 0.0)

    import math

    # 전체 모듈 면적 (base_scale)
    base_scale = sum(m.area for m in modules)
    # 네트워크에 연결된 모듈들의 면적 합 (HPWL 정규화에 사용)
    net_area_sum = sum(m.area for m in modules if m.net)
    if net_area_sum == 0:
        net_area_sum = base_scale

    # 전체 배치의 bounding box 계산
    _, _, area_bb = calculate_total_area(modules)
    # HPWL 계산
    hpwl = calculate_hpwl(modules)

    # --- 패널티 계산 ---
    if chip:
        bound_w = chip.bound.width
        bound_h = chip.bound.height

        # 전체 배치의 bounding box (w_bb, h_bb) 계산
        w_bb, h_bb, _ = calculate_total_area(modules)

        # [Area Violation Function (A′)]
        if w_bb <= bound_w and h_bb <= bound_h:
            area_violation = 0.0
        elif w_bb > bound_w and h_bb <= bound_h:
            area_violation = (w_bb - bound_w) * bound_h
        elif w_bb <= bound_w and h_bb > bound_h:
            area_violation = (h_bb - bound_h) * bound_w
        else:
            area_violation = (w_bb * h_bb) - (bound_w * bound_h)

        # [Excessive Length Violation Function (L′)]
        length_violation = 0.0
        for mm in modules:
            over_w = (mm.x + mm.width) - bound_w
            over_h = (mm.y + mm.height) - bound_h
            if over_w > 0:
                length_violation += over_w ** 2
            if over_h > 0:
                length_violation += over_h ** 2

        penalty_sum = area_violation + length_violation
    else:
        penalty_sum = 0.0

    # 정규화
    area_norm = (area_bb / base_scale) if base_scale else 0
    hpwl_norm = hpwl / (2 * math.sqrt(net_area_sum)) if net_area_sum > 0 else 0
    # 패널티 정규화: base_scale을 기준으로 정규화 (0이 아닌 문제 스케일 사용)
    penalty_norm = penalty_sum / base_scale if base_scale else 0


    # area_norm과 HPWL_norm의 스케일 차를 줄이기 위해 area_norm에 500 곱함
    area_norm *= 500
    # penalty_norm scale 조절
    penalty_norm *= 50

    return (area_norm, hpwl_norm, penalty_norm)



# ─────────────────────────────────────────────────────────────
# 3. FastSA 함수 (sample_moves로 평균 dE 구해 T 업데이트 후,
#    그 sample 중 제일 좋은 해만 "동일한 연산"으로 다시 적용 + Acceptance.)
# ─────────────────────────────────────────────────────────────

def fast_sa(chip, max_iter=50, P=0.99, c=100, w=0.5, sample_moves=10, r=1.0):
    import copy, math, random

    T1_scale_factor = 10.0

    # SA 시작 전 기록용 리스트 초기화 (Penalty 변화 모니터링)
    iter_history = []
    penalty_history = []

    # 초기 상태 백업
    orig_st = copy.deepcopy(chip)
    orig_cost = calc_combined_cost(chip.modules, w, chip=chip, r=r)

    # 1) up_diffs 수집 -> T1 추정
    up_diffs = []
    for _ in range(sample_moves + 10):
        msg, op, opdata = chip.apply_random_operation()
        new_c = calc_combined_cost(chip.modules, w, chip=chip, r=r)
        dE = new_c - orig_cost
        if dE > 0:
            up_diffs.append(dE)
        chip = copy.deepcopy(orig_st)

    if up_diffs:
        delta_avg = sum(up_diffs) / len(up_diffs)
    else:
        delta_avg = 1.0
    if delta_avg < 1e-12:
        delta_avg = 1.0

    T1 = abs(delta_avg / math.log(P)) * T1_scale_factor
    print(f"Initial T1={T1:.3f}")

    best_chip = copy.deepcopy(chip)
    best_cost = orig_cost
    cur_cost = orig_cost

    temps = []
    # ─────────────────────────────────────────────
    # 메인 SA 루프
    for n in range(1, max_iter + 1):
        # (a) 현재 해 백업
        st_cp = copy.deepcopy(chip)
        oldc = cur_cost

        # (b) sample_moves번 "임시"연산, 그 중 최고의 해를 찾기 (op_data 저장)
        cost_diffs = []
        best_local_cost = float('inf')
        best_op_data = None  # op_data dict
        best_msg = None  # not used for reapplication, just for logs

        for _ in range(sample_moves):
            msg_tmp, op_tmp, op_data_tmp = chip.apply_random_operation()
            tmp_cost = calc_combined_cost(chip.modules, w, chip=chip, r=r)
            dE_ = tmp_cost - oldc
            cost_diffs.append(abs(dE_))

            # 이 sample 내에서 최적 cost 갱신
            if tmp_cost < best_local_cost:
                best_local_cost = tmp_cost
                best_op_data = op_data_tmp
                best_msg = msg_tmp

            # 복귀 (sample 용이므로 실제로 적용 X)
            chip = copy.deepcopy(st_cp)

        # (c) sample_moves로부터 delta_cost 계산 -> T 업데이트
        if cost_diffs:
            delta_cost = max(sum(cost_diffs) / len(cost_diffs), 1e-6)
        else:
            delta_cost = 1e-6

        if n == 1:
            T = T1
        elif 2 <= n <= 400:
            T = max((T1 * delta_cost) / (n * c), 1e-6)
        else:
            T = max((T1 * delta_cost) / n, 1e-6)
        temps.append(T)

        # (d) 최고의 op_data를 실제로 재적용
        old_chip = copy.deepcopy(chip)
        old_cost = cur_cost
        re_msg, re_op = chip.reapply_operation(best_op_data)  # 정확히 같은 노드/연산

        new_c = calc_combined_cost(chip.modules, w, chip=chip, r=r)
        dE = new_c - old_cost

        # (e) Acceptance
        if dE < 0:
            cur_cost = new_c
            if new_c < best_cost:
                best_cost = new_c
                best_chip = copy.deepcopy(chip)
            acc_str = "ACCEPT (better)"
            acc_prob = 1.0
        else:
            if T < 1e-12:
                acc_prob = 0.0
            else:
                acc_prob = math.exp(-abs(dE) / T)
            if random.random() < acc_prob:
                cur_cost = new_c
                if new_c < best_cost:
                    best_cost = new_c
                    best_chip = copy.deepcopy(chip)
                acc_str = "ACCEPT (worse)"
            else:
                # 거부 -> 원상 복귀
                chip = copy.deepcopy(old_chip)
                cur_cost = old_cost
                acc_str = "REJECT"

        print(f"[Iter={n:3d}] BestLocalOpData={best_op_data}, BestLocalCost={best_local_cost:.3f}, "
              f"ReAppMsg={re_msg}, T={T:9.5f}, dE={dE:9.5f}, Prob={acc_prob:6.4f}, {acc_str}")

        # (f) n%4000 == 0 시점 plot (사용자가 3/8에 5000번마다 plot이라 하였으나
        #   본 예시에서는 너무 커서 40000등으로 조건만 변경해둠)
        # if n % 10000 == 0:
        #    best_chip.plot_b_tree(iteration=n)
        #    plt.show()

        ##(추가)
        if n % 5000 == 0:
            # --- 현재 비용 계산 ---
            current_cost = calc_combined_cost(chip.modules, w, chip=chip, r=r)
            # get_normalized_values()는 area_norm, hpwl_norm, penalty_norm을 반환합니다.
            area_norm, hpwl_norm, penalty_norm = get_normalized_values(chip.modules, chip=chip, w=w, r=r)


            # --- 현재 지표 계산 ---
            hpwl_val = calculate_hpwl(chip.modules)
            w_bb, h_bb, area_bb = calculate_total_area(chip.modules)
            total_module_area = sum(m.width * m.height for m in chip.modules)
            dead_space_rate_val = ((area_bb - total_module_area) / area_bb * 100) if area_bb > 0 else 0.0


            # --- 기록 업데이트 ---
            iter_history.append(n)
            penalty_history.append(penalty_norm)

            # --- 시각화용 플롯 생성 (3 서브플롯: floorplan, B*-Tree) ---
            import matplotlib.pyplot as plt
            import networkx as nx

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # (1) Floorplan 시각화
            ax1.set_title("B*-Tree Placement (Contour+DFS)")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_xlim(0, chip.bound.width + 50)
            ax1.set_ylim(0, chip.bound.height + 50)
            ax1.set_aspect('equal')
            ax1.grid(True, linestyle='--', alpha=0.5)

            for m in chip.modules:
                rect = plt.Rectangle((m.x, m.y), m.width, m.height,
                                     edgecolor='blue', facecolor='skyblue', alpha=0.6)
                ax1.add_patch(rect)
                ax1.text(m.x + m.width / 2, m.y + m.height / 2, m.name,
                         ha='center', va='center', fontsize=8)

            # 빨간 경계선 (outline)
            br = plt.Rectangle((0, 0), chip.bound.width, chip.bound.height,
                               edgecolor='red', facecolor='none', lw=2)
            ax1.add_patch(br)

            # 지표 텍스트 박스
            metrics_text = (f"Iteration: {n}\n"
                            f"Area_norm: {area_norm:.2f}\n"
                            f"HPWL_nrom: {hpwl_norm:.2f}\n"
                            f"Penalty_norm: {penalty_norm:.2f}\n"
                            f"Cost: {current_cost:.2f}\n"
                            f"Dead Space: {dead_space_rate_val:.2f}%")
            ax1.text(0.95, 0.95, metrics_text, transform=ax1.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.7))

            # (2) B*-Tree 구조 시각화
            G = nx.DiGraph()
            pos = {}

            def add_tree_edges(node, parent=None, depth=0, x_pos=0):
                if node is None:
                    return
                G.add_node(node.module.name)
                pos[node.module.name] = (x_pos, -depth)
                if parent:
                    G.add_edge(parent.module.name, node.module.name)
                add_tree_edges(node.left, node, depth + 1, x_pos - 1)
                add_tree_edges(node.right, node, depth + 1, x_pos + 1)

            add_tree_edges(chip.root)
            nx.draw(G, pos, with_labels=True, node_color="lightblue",
                    edge_color="gray", node_size=1200, font_size=9, ax=ax2)
            ax2.set_title("B*-Tree Structure")

            plt.tight_layout()
            plt.show(block=False)  # 플롯을 비블로킹으로 표시
            plt.close()

    # ─────────────────────────────────────────────────────────────
    # 마무리: T vs Iter 플롯
    plt.figure()
    plt.plot(range(1, max_iter + 1), temps)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.title("Temperature vs. Iteration")
    plt.show()

    ##(추가) 최종 Penalty Evolution 플롯 추가
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(iter_history, penalty_history, marker='o', linestyle='-', color='tab:red')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Penalty")
    ax.set_title("Penalty Evolution (Final)")

    plt.tight_layout()
    plt.show()  # 마지막에는 blocking 모드로 플롯을 띄움

    return best_chip


# ─────────────────────────────────────────────────────────────
# 4. 파일 파싱(Yal, GSRC)
# ─────────────────────────────────────────────────────────────

def parse_yal(file_path):
    modules = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    module_data = {}
    in_network = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("/*") or line.startswith("*"):
            continue
        if line.startswith("MODULE"):
            module_name = line.split()[1].strip(';')
            module_data = {'name': module_name, 'net': []}
            continue
        if line.startswith("TYPE"):
            module_type = line.split()[1].strip(';')
            module_data['type'] = module_type
            continue
        if line.startswith("DIMENSIONS"):
            dims = list(map(float, line.replace("DIMENSIONS", "").replace(";", "").split()))
            x_coords = [dims[i] for i in range(0, len(dims), 2)]
            y_coords = [dims[i + 1] for i in range(0, len(dims), 2)]
            module_data['width'] = max(x_coords) - min(x_coords)
            module_data['height'] = max(y_coords) - min(y_coords)
            continue
        if line.startswith("NETWORK"):
            in_network = True
            continue
        if line.startswith("ENDNETWORK"):
            in_network = False
            continue
        if line.startswith("ENDMODULE"):
            modules.append(Module(
                name=module_data['name'],
                width=module_data['width'],
                height=module_data['height'],
                module_type=module_data['type'],
                net=module_data['net']
            ))
            module_data = {}
            continue
        if in_network:
            module_data['net'].append(line.strip(';'))
            continue
    return modules


def parse_gsrc_blocks(blocks_file):
    """
    Terminal( pX )은 무시하고, sbX 블록만 파싱
    """
    modules = []
    pattern = re.compile(r"^(sb\d+)\s+\S+\s+(\d+)\s+(.*)$")
    with open(blocks_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('UCSC'):
            continue

        # 만약 p1 terminal 나오면 => 그냥 skip
        if line.startswith('p') and 'terminal' in line:
            continue

        match = pattern.match(line)
        if match:
            blk_name = match.group(1)  # sb0, sb1 ...
            coords = match.group(3)  # (0,0) ...
            cpat = re.compile(r"\(([\-\d\.]+)\s*,\s*([\-\d\.]+)\)")
            found = cpat.findall(coords)
            xs, ys = [], []
            for (sx, sy) in found:
                xs.append(float(sx))
                ys.append(float(sy))
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            w_ = maxx - minx
            h_ = maxy - miny
            modules.append(Module(blk_name, w_, h_, 'BLOCK', net=[]))

    return modules


def parse_gsrc_nets(nets_file, modules):
    """
    pX 등 terminal은 이미 parse_gsrc_blocks에서 안 넣었으므로,
    name_map에 없는 pin_name => 무시됨
    """
    name_map = {m.name: m for m in modules}
    i = 0
    with open(nets_file, 'r') as f:
        lines = f.readlines()
    net_id = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#') or line.startswith('UCLA'):
            i += 1
            continue
        if line.startswith('NetDegree'):
            parts = line.split(':')
            deg_str = (parts[1].strip() if len(parts) > 1 else "0")
            try:
                deg = int(deg_str)
            except:
                deg = 0
            net_name = f"Net{net_id}"
            net_id += 1
            pins = []
            for _ in range(deg):
                i += 1
                pin_ln = lines[i].strip()
                pin_prt = pin_ln.split()
                if pin_prt:
                    pin_name = pin_prt[0]
                    # pX 는 name_map에 없으므로, 자동 무시
                    if pin_name in name_map:
                        pins.append(pin_name)
            for pn in pins:
                name_map[pn].net.append(net_name)
        i += 1
    return modules


# ─────────────────────────────────────────────────────────────
# 5. 메인 실행 예시 (Left-Chain + 노드 간격/크기 조정, 서브트리 SA 제거)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # (1) Yal 파일 파싱
    # yal_file="./example/ami49.yal"
    # modules=parse_yal(yal_file)

    # (2) GSRC blocks & nets 파싱 (터미널 무시)
    blocks_file = "C:/Users/82109/n200.blocks"
    nets_file = "C:/Users/82109/n200.nets"
    modules = parse_gsrc_blocks(blocks_file)
    modules = parse_gsrc_nets(nets_file, modules)

    chip = Chip(modules)
    chip.calculate_coordinates()
    chip.plot_b_tree()

    # 초기 상태에서의 area, HPWL, penalty, cost 계산 (정규화된 값)
    w_, h_, area_ = calculate_total_area(chip.modules)
    hpwl_ = calculate_hpwl(chip.modules)
    init_cost = calc_combined_cost(chip.modules, w=0.66, chip=chip, r=1.0)
    area_norm, hpwl_norm, penalty_norm = get_normalized_values(chip.modules, chip=chip, w=0.66, r=1.0)

    print("=== 초기 Chip (Left-Chain) ===")
    print(f"BoundingBox area (absolute)   = {area_}")
    print(f"HPWL (absolute)              = {hpwl_}")
    print(f"Area_norm                    = {area_norm:.3f}")
    print(f"HPWL_norm                    = {hpwl_norm:.3f}")
    print(f"Penalty_norm                 = {penalty_norm:.3f}")
    print(f"Initial cost (with penalty)  = {init_cost:.3f}")

    # base_scale = sum(m.area for m in modules)
    # print(f"{base_scale :.3f}")


    ans = input("FastSA로 최적화를 진행?(y/n): ")
    if ans.lower().startswith('y'):
        best_chip = fast_sa(
            chip,
            max_iter=50000,
            P=0.95,
            c=20,
            w=0.66,
            sample_moves=30,
            r=1.0  # penalty 가중치
        )

        # 최종 상태에서의 area, HPWL, penalty, cost 계산 (정규화된 값)
        final_w, final_h, final_area = calculate_total_area(best_chip.modules)
        final_hpwl = calculate_hpwl(best_chip.modules)
        final_cost = calc_combined_cost(best_chip.modules, w=0.5, chip=best_chip, r=1.0)
        f_area_norm, f_hpwl_norm, f_penalty_norm = get_normalized_values(best_chip.modules, best_chip, w=0.5, r=1.0)

        total_module_area = sum(m.area for m in best_chip.modules)
        if final_area > 0:
            ds = ((final_area - total_module_area) / final_area) * 100
        else:
            ds = 0.0

        print("=== FastSA 종료 ===")
        print(f"최종 BoundingBox area (absolute)   = {final_area}")
        print(f"최종 HPWL (absolute)              = {final_hpwl}")
        print(f"최종 Area_norm                    = {f_area_norm:.3f}")
        print(f"최종 HPWL_norm                    = {f_hpwl_norm:.3f}")
        print(f"최종 Penalty_norm                 = {f_penalty_norm:.3f}")
        print(f"최종 cost (with penalty)          = {final_cost:.3f}")
        print(f"DeadSpaceRate                     = {ds:.3f}%")

        best_chip.plot_b_tree()
        plt.show()