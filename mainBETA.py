import matplotlib.pyplot as plt
import networkx as nx
import random

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
        self.order = None  # 순서 번호 필드

    def rotate(self):
        """너비와 높이를 교환"""
        self.width, self.height = self.height, self.width

    def set_position(self, x: float, y: float, order: int):
        self.x = x
        self.y = y
        self.order = order

    def __str__(self):
        formatted_net = "\n    ".join(self.net)
        return (f"Module(Name={self.name}, Width={self.width:.2f}, Height={self.height:.2f}, "
                f"Area={self.area:.2f}, Position=({self.x:.2f}, {self.y:.2f}), "
                f"Order={self.order}, Type={self.type}, Net=\n    {formatted_net})")

class BTreeNode:
    def __init__(self, module, parent=None):
        self.module = module
        self.left = None
        self.right = None
        self.parent = parent

class Chip:
    def __init__(self, modules, default_width=1000, default_height=1000):
        self.modules = [m for m in modules if m.type != 'PARENT']
        self.bound = next((m for m in modules if m.type == 'PARENT'), None)
        if not self.bound:
            self.bound = Module(name='DefaultParent', width=default_width, height=default_height, module_type='PARENT')
        self.build_b_tree()

    def build_b_tree(self):
        """간단한 완전 이진 트리 형태로 B*-Tree를 구성"""
        if not self.modules:
            self.root = None
            return

        # 가장 앞의 모듈을 루트로
        self.root = BTreeNode(self.modules[0])
        queue = [self.root]

        for module in self.modules[1:]:
            parent = queue[0]
            if parent.left is None:
                parent.left = BTreeNode(module, parent)
                queue.append(parent.left)
            elif parent.right is None:
                parent.right = BTreeNode(module, parent)
                queue.append(parent.right)
                queue.pop(0)

    # ─────────────────────────────────────────
    # B*-Tree 구조에서 모든 노드를 수집 (BFS)
    # ─────────────────────────────────────────
    def collect_all_nodes(self):
        if not self.root:
            return []
        result = []
        queue = [self.root]
        while queue:
            curr = queue.pop(0)
            result.append(curr)
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        return result

    # ─────────────────────────────────────────
    # Op1: Rotate (회전) - 구조 변화 없음
    # ─────────────────────────────────────────
    def rotate_node(self, node):
        node.module.rotate()

    # ─────────────────────────────────────────
    # Op2: Move (노드를 트리 내 다른 자리로 이동)
    # ─────────────────────────────────────────
    def move_node(self, node):
        """
        1) node가 루트면 이동 불가(간단 처리)
        2) node를 부모에서 떼어냄
        3) 트리 전체를 순회해 "왼/오 자식 슬롯이 비어있는 노드들"을 찾음
        4) 무작위 후보 부모를 골라 왼 또는 오른쪽에 붙임
        """
        if node.parent is None:
            return f"[Op2: Move] Cannot move ROOT node ({node.module.name})."

        # (1) 부모 포인터 끊기
        self.remove_node_from_parent(node)

        # (2) 가능 후보 수집
        candidates = self.find_possible_parents()
        if not candidates:
            return "[Op2: Move] No available spot to re-insert node."

        new_parent = random.choice(candidates)
        side = self.attach_node(new_parent, node)
        return f"[Op2: Move] Moved {node.module.name} under {new_parent.module.name} ({side})"

    def remove_node_from_parent(self, node):
        """현재 node를 부모의 left/right에서 제거"""
        p = node.parent
        if p is None:
            return
        if p.left == node:
            p.left = None
        elif p.right == node:
            p.right = None
        node.parent = None

    def find_possible_parents(self):
        """BFS: 왼쪽 또는 오른쪽 자식이 None인 노드들 수집"""
        all_nodes = self.collect_all_nodes()
        results = []
        for nd in all_nodes:
            # 하나라도 None이면 candidate
            if nd.left is None or nd.right is None:
                results.append(nd)
        return results

    def attach_node(self, parent, node):
        """parent의 비어있는(left/right) 자리 중 하나에 node를 붙임"""
        node.parent = parent
        empty_slots = []
        if parent.left is None:
            empty_slots.append("left")
        if parent.right is None:
            empty_slots.append("right")
        if not empty_slots:
            return "NoSlot"
        chosen = random.choice(empty_slots)
        if chosen == "left":
            parent.left = node
        else:
            parent.right = node
        return chosen

    # ─────────────────────────────────────────
    # Op3: Swap (두 노드의 module 교환)
    # ─────────────────────────────────────────
    def swap_nodes(self, nodeA, nodeB):
        nodeA.module, nodeB.module = nodeB.module, nodeA.module

    # ─────────────────────────────────────────
    # 무작위 연산 적용 (Op1/Op2/Op3)
    # ─────────────────────────────────────────
    def apply_random_operation(self):
        ops = ["rotate", "move", "swap"]
        op = random.choice(ops)

        all_nodes = self.collect_all_nodes()
        if not all_nodes:
            print("[Warn] No nodes in the tree.")
            return

        # 연산 적용
        if op == "rotate":
            chosen = random.choice(all_nodes)
            self.rotate_node(chosen)
            msg = f"[Op1: Rotate] Rotated {chosen.module.name}"

        elif op == "move":
            chosen = random.choice(all_nodes)
            msg = self.move_node(chosen)

        else:  # swap
            if len(all_nodes) < 2:
                msg = "[Op3: Swap] Not enough nodes to swap."
            else:
                nodeA, nodeB = random.sample(all_nodes, 2)
                self.swap_nodes(nodeA, nodeB)
                msg = f"[Op3: Swap] Swapped {nodeA.module.name} <-> {nodeB.module.name}"

        # ── 콘솔 출력 먼저 ──
        print(msg)

        # ── 좌표 리셋 → 새로 배치, 플롯 ──
        for nd in all_nodes:
            nd.module.x = 0
            nd.module.y = 0
            nd.module.order = None

        self.calculate_coordinates()
        self.plot_b_tree()

    # ─────────────────────────────────────────
    # B*-Tree 좌표 배치
    # ─────────────────────────────────────────
    def calculate_coordinates(self, node=None, x_offset=0, y_offset=0, placed_modules=None, order=1):
        """B*-Tree 규칙대로 좌표 배치 + adjust_for_overlaps로 충돌 회피"""
        if self.root is None:
            return
        if node is None:
            node = self.root
        if placed_modules is None:
            placed_modules = []

        if node.parent is None:
            x, y = x_offset, y_offset
        else:
            # 왼쪽 자식 -> 부모 모듈의 (x+width)
            if node == node.parent.left:
                x = node.parent.module.x + node.parent.module.width
                y = node.parent.module.y
            # 오른쪽 자식 -> 부모 모듈의 (y+height)
            else:
                x = node.parent.module.x
                y = node.parent.module.y + node.parent.module.height

        x, y = self.adjust_for_overlaps(node.module, x, y, placed_modules)
        node.module.set_position(x, y, order)
        placed_modules.append(node.module)

        if node.left:
            order = self.calculate_coordinates(node.left, x, y, placed_modules, order+1)
        if node.right:
            order = self.calculate_coordinates(node.right, x, y, placed_modules, order+1)
        return order

    def adjust_for_overlaps(self, module, x_start, y_start, placed_modules):
        """
        간단한 충돌 회피:
         - (x_start, y_start)에서 시작해, 겹침 시 x += 1
         - bound 넘어가면 x=0, y += 1
         - 전부 넘어가면 예외
        """
        x, y = x_start, y_start
        while True:
            if self.is_within_bounds(module, x, y) and not self.check_overlap(module, x, y, placed_modules):
                return x, y
            x += 1
            # 줄 바꿈
            if x + module.width > self.bound.width:
                x = 0
                y += 1
            if y + module.height > self.bound.height:
                raise ValueError("Unable to fit module within bounds")

    def check_overlap(self, module, x, y, placed_modules):
        for placed in placed_modules:
            if (x < placed.x + placed.width and
                x + module.width > placed.x and
                y < placed.y + placed.height and
                y + module.height > placed.y):
                return True
        return False

    def is_within_bounds(self, module, x, y):
        return (x + module.width <= self.bound.width) and (y + module.height <= self.bound.height)

    # ─────────────────────────────────────────
    # 시각화
    # ─────────────────────────────────────────
    def plot_b_tree(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        # (1) 물리적 배치도
        self._plot_node(self.root, ax1)
        bound_rect = plt.Rectangle((0, 0), self.bound.width, self.bound.height,
                                   edgecolor='red', facecolor='none', lw=2)
        ax1.add_patch(bound_rect)
        ax1.set_title("B*-Tree Physical Placement")
        ax1.set_xlabel("X-coordinate")
        ax1.set_ylabel("Y-coordinate")
        ax1.set_xlim(0, self.bound.width + 100)
        ax1.set_ylim(0, self.bound.height + 100)

        # (2) B*-Tree 구조
        G = nx.DiGraph()
        pos = {}
        self.add_edges(self.root, None, 0, 0, G, pos)
        nx.draw(G, pos, ax=ax2, with_labels=True, node_color="lightblue",
                edge_color="gray", node_size=2000, font_size=10)
        ax2.set_title("B*-Tree Structure")

        plt.show()

    def _plot_node(self, node, ax):
        if not node:
            return
        module = node.module
        rect = plt.Rectangle((module.x, module.y), module.width, module.height,
                             edgecolor='blue', facecolor='lightblue', fill=True, lw=2)
        ax.add_patch(rect)
        ax.text(module.x + module.width/2, module.y + module.height/2,
                f'{module.name}\n#{module.order}', ha='center', va='center', fontsize=8)

        self._plot_node(node.left, ax)
        self._plot_node(node.right, ax)

    def add_edges(self, node, parent_name, depth, x_pos, G, pos):
        if not node:
            return x_pos
        node_name = node.module.name
        pos[node_name] = (x_pos, -depth)
        G.add_node(node_name, label=node.module.name)
        if parent_name:
            G.add_edge(parent_name, node_name)

        x_pos = self.add_edges(node.left, node_name, depth+1, x_pos, G, pos)
        x_pos = self.add_edges(node.right, node_name, depth+1, x_pos+1, G, pos)
        return x_pos

# ───────────────────────────────────────────────────────────────────
# YAL 파싱 (기존과 동일)
# ───────────────────────────────────────────────────────────────────
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
            dimensions = list(map(float, line.replace("DIMENSIONS", "").replace(";", "").split()))
            x_coords = [dimensions[i] for i in range(0, len(dimensions), 2)]
            y_coords = [dimensions[i + 1] for i in range(0, len(dimensions), 2)]
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

# ───────────────────────────────────────────────────────────────────
# 메인 실행 예시
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    yal_file = "./example/gptexample.yal"  # YAL 파일 경로
    modules = parse_yal(yal_file)
    chip = Chip(modules)

    # 초기 한번 배치 & 플롯
    chip.calculate_coordinates()
    chip.plot_b_tree()

    while True:
        command = input("Enter 를 눌러 (Op1/Op2/Op3) 무작위 연산, '1'을 입력하면 종료: ")
        if command.strip() == '1':
            print("Exiting...")
            break
        chip.apply_random_operation()
