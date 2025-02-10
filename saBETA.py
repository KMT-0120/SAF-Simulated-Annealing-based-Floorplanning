import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import random

# ─────────────────────────────────────────────────────────────
# 1. Module, BTreeNode, Chip 클래스
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
    """
    B*-Tree 기반 배치를 관리하는 클래스.
    - modules: 배치할 모듈들
    - bound: 부모로 쓰이는 '칩' 영역 모듈
    """
    def __init__(self, modules, default_width=1000, default_height=1000):
        self.modules = [m for m in modules if m.type != 'PARENT']
        self.bound = next((m for m in modules if m.type == 'PARENT'), None)
        if not self.bound:
            self.bound = Module(
                name='DefaultParent',
                width=default_width,
                height=default_height,
                module_type='PARENT'
            )
        self.build_b_tree()

    # ─────────────────────────────────────────
    # B*-Tree 구성 (간단한 완전 이진 트리 형태)
    # ─────────────────────────────────────────
    def build_b_tree(self):
        if not self.modules:
            self.root = None
            return
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

    def collect_all_nodes(self):
        """BFS로 모든 노드 수집"""
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
    # Op1: Rotate
    # ─────────────────────────────────────────
    def rotate_node(self, node):
        node.module.rotate()

    # ─────────────────────────────────────────
    # Op2: Move
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
        """node를 현재 부모의 left/right에서 제거"""
        p = node.parent
        if p is None:
            return
        if p.left == node:
            p.left = None
        elif p.right == node:
            p.right = None
        node.parent = None

    def find_possible_parents(self):
        """BFS로 수집: 왼쪽/오른쪽 자식 중 비어있는 노드를 후보로"""
        all_nodes = self.collect_all_nodes()
        results = []
        for nd in all_nodes:
            if nd.left is None or nd.right is None:
                results.append(nd)
        return results

    def attach_node(self, parent, node):
        """parent 노드의 비어있는 (left/right) 중 하나에 node 붙임"""
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
    # Op3: Swap
    # ─────────────────────────────────────────
    def swap_nodes(self, nodeA, nodeB):
        nodeA.module, nodeB.module = nodeB.module, nodeA.module

    # ─────────────────────────────────────────
    # 무작위 연산 (Op1/Op2/Op3)
    # ─────────────────────────────────────────
    def apply_random_operation(self):
        ops = ["rotate", "move", "swap"]
        op = random.choice(ops)

        all_nodes = self.collect_all_nodes()
        if not all_nodes:
            print("[Warn] No nodes in the tree.")
            return ("NoNode", None)

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

        print(msg)

        # 좌표 리셋 후 새로 배치
        for nd in all_nodes:
            nd.module.x = 0
            nd.module.y = 0
            nd.module.order = None
        self.calculate_coordinates()

        return (msg, op)

    def calculate_coordinates(self, node=None, x_offset=0, y_offset=0, placed_modules=None, order=1):
        """
        B*-Tree 규칙대로 좌표 배치 + adjust_for_overlaps로 간단 충돌 회피
        """
        if self.root is None:
            return
        if node is None:
            node = self.root
        if placed_modules is None:
            placed_modules = []

        if node.parent is None:
            x, y = x_offset, y_offset
        else:
            # 왼쪽 자식 = (부모.x + 부모.width, 부모.y)
            if node == node.parent.left:
                x = node.parent.module.x + node.parent.module.width
                y = node.parent.module.y
            else:  # 오른쪽 자식 = (부모.x, 부모.y + 부모.height)
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
        충돌 회피(간단 버전):
         - (x_start, y_start)에서 시작해 겹치면 x += 1, 범위 초과 시 줄바꿈
        """
        x, y = x_start, y_start
        while self.check_overlap(module, x, y, placed_modules):
            x += 1
            if x + module.width > self.bound.width:
                x = 0
                y += 1
        return x, y

    def check_overlap(self, module, x, y, placed_modules):
        for placed in placed_modules:
            overlap = (x < placed.x + placed.width and
                       x + module.width > placed.x and
                       y < placed.y + placed.height and
                       y + module.height > placed.y)
            if overlap:
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

        # (2) B*-Tree 구조 (그래프)
        G = nx.DiGraph()
        pos = {}
        self.add_edges(self.root, None, 0, 0, G, pos)
        nx.draw(
            G, pos, ax=ax2, with_labels=True, node_color="lightblue",
            edge_color="gray", node_size=2000, font_size=10
        )
        ax2.set_title("B*-Tree Structure")

    def _plot_node(self, node, ax):
        if not node:
            return
        module = node.module
        rect = plt.Rectangle(
            (module.x, module.y), module.width, module.height,
            edgecolor='blue', facecolor='lightblue', fill=True, lw=2
        )
        ax.add_patch(rect)
        ax.text(
            module.x + module.width/2, module.y + module.height/2,
            f'{module.name}\n#{module.order}', ha='center', va='center', fontsize=8
        )

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

    # ─────────────────────────────────────────
    # 무작위로 "modules"를 섞어서 새로운 배치를 만든 뒤 Cost 계산
    # ─────────────────────────────────────────
    def randomize_b_tree(self, w=0.5):
        random.shuffle(self.modules)
        self.build_b_tree()
        self.calculate_coordinates()
        return calc_combined_cost(self.modules, w)


# ─────────────────────────────────────────────────────────────
# 2. Cost 계산 함수들 (간단 정규화)
# ─────────────────────────────────────────────────────────────
def calculate_hpwl(modules):
    """
    HPWL = (max_x - min_x) + (max_y - min_y)
    """
    if not modules:
        return 0
    min_x = min(m.x for m in modules)
    max_x = max(m.x + m.width for m in modules)
    min_y = min(m.y for m in modules)
    max_y = max(m.y + m.height for m in modules)
    return (max_x - min_x) + (max_y - min_y)

def calculate_total_area(modules):
    """
    bounding box의 폭, 높이, 면적 반환
    """
    if not modules:
        return 0,0,0
    min_x = min(m.x for m in modules)
    max_x = max(m.x + m.width for m in modules)
    min_y = min(m.y for m in modules)
    max_y = max(m.y + m.height for m in modules)
    width = (max_x - min_x)
    height = (max_y - min_y)
    return width, height, (width * height)

def calc_combined_cost(modules, w=0.5):
    """
    간단 정규화:
      area_bb/base_scale + hpwl/hpwl_scale
    """
    if not modules:
        return 0.0

    hpwl_scale = 10.0   # HPWL 정규화 상수
    base_scale = 150.0  # area 정규화 상수

    _, _, area_bb = calculate_total_area(modules)
    hpwl = calculate_hpwl(modules)

    area_norm = area_bb / base_scale
    hpwl_norm = hpwl / hpwl_scale

    return w * area_norm + (1 - w) * hpwl_norm


# ─────────────────────────────────────────────────────────────
# 3. FastSA 함수 (수정된 \Delta_avg, \Delta_cost 정의 적용)
# ─────────────────────────────────────────────────────────────
def fast_sa(chip, max_iter=50, P=0.99, c=100, w=0.5, sample_moves=10):
    """
    FastSA + 간단 정규화 cost
      - \Delta_avg : 초기 온도 설정 시, sample_moves 번 랜덤 move에서
                     오직 "uphill" (cost 증가)인 move만 평균낸 값
      - \Delta_cost: 각 iteration 마다, 현 온도에서 여러 번(move) 시도해본
                     cost 변화량(절댓값)들의 평균
    """
    import copy, math, random

    # (1) 초기 온도 설정을 위한 \Delta_avg 측정
    original_state = copy.deepcopy(chip)
    original_cost = calc_combined_cost(chip.modules, w)

    uphill_diffs = []
    for _ in range(sample_moves):
        msg, op = chip.apply_random_operation()
        new_cost = calc_combined_cost(chip.modules, w)
        delta_e = new_cost - original_cost
        # 올라간(Uphill) move만 수집
        if delta_e > 0:
            uphill_diffs.append(delta_e)
        # 롤백
        chip = copy.deepcopy(original_state)

    # 올라간 move가 하나도 없으면 1.0 사용
    if uphill_diffs:
        delta_avg = sum(uphill_diffs) / len(uphill_diffs)
    else:
        delta_avg = 1.0
    if delta_avg < 1e-12:
        delta_avg = 1.0

    # 초기 온도 T1 = delta_avg / ln(P) * a (a는 보정계수)
    T1_scale_factor = 0.3
    T1 = abs(delta_avg / math.log(P)) * T1_scale_factor
    print(f"Initial T1 = {T1:.3f}")

    # (2) 초기 cost, best cost
    best_chip = copy.deepcopy(chip)
    best_cost = original_cost
    current_cost = original_cost

    # (3) SA 반복
    for n in range(1, max_iter+1):

        # ── 이번 iteration에서의 \Delta_cost (평균 cost 변화량) 계산 ──
        #    현재 상태에서 sample_moves번 move를 해보고 그 절댓값을 평균
        state_copy = copy.deepcopy(chip)
        cost_diffs_iter = []
        for _ in range(sample_moves):
            msg_tmp, op_tmp = chip.apply_random_operation()
            tmp_cost = calc_combined_cost(chip.modules, w)
            cost_diffs_iter.append(abs(tmp_cost - current_cost))
            # 매번 롤백
            chip = copy.deepcopy(state_copy)

        if cost_diffs_iter:
            delta_cost = max(sum(cost_diffs_iter) / len(cost_diffs_iter), 1e-6)
        else:
            delta_cost = 1e-6

        # ── 실제로 1번 move를 진행해서 새로운 해 후보를 만든다 ──
        old_chip_state = copy.deepcopy(chip)
        old_cost = current_cost

        msg, op = chip.apply_random_operation()
        new_cost = calc_combined_cost(chip.modules, w)
        delta_e = new_cost - old_cost

        # 온도 T_n
        if n == 1:
            T = T1
        elif 2 <= n <= 7:
            T = max((T1 * delta_cost) / (n * c), 1e-6)
        else:
            T = max((T1 * delta_cost) / n, 1e-6)

        # SA 수용 여부(메트로폴리스 기준)
        if delta_e < 0:
            # 개선된 해(비용 감소)
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_chip = copy.deepcopy(chip)
            accept_str = "ACCEPT (better)"
            accept_prob = 1.0
        else:
            # 악화된 해(비용 증가)
            if T < 1e-12:
                accept_prob = 0.0
            else:
                accept_prob = math.exp(-abs(delta_e) / T)

            if random.random() < accept_prob:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_chip = copy.deepcopy(chip)
                accept_str = "ACCEPT (worse)"
            else:
                # 거부 -> 롤백
                chip = copy.deepcopy(old_chip_state)
                current_cost = old_cost
                accept_str = "REJECT"

        print(f"[Iter={n:3d}] Op={op.upper():5s}, "
              f"T={T:9.5f}, ΔE={delta_e:9.5f}, "
              f"deltaCost={delta_cost:9.5f}, "
              f"Prob={accept_prob:6.4f}, {accept_str}")

    return best_chip


# ─────────────────────────────────────────────────────────────
# 4. YAL 파일 파싱
# ─────────────────────────────────────────────────────────────
def parse_yal(file_path):
    """
    YAL 파일을 읽어 Module 객체 리스트로 반환
    """
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
            dimensions = list(map(float, line.replace("DIMENSIONS", "")
                                                  .replace(";", "").split()))
            x_coords = [dimensions[i] for i in range(0, len(dimensions), 2)]
            y_coords = [dimensions[i+1] for i in range(0, len(dimensions), 2)]
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
            # 모듈 완성
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


# ─────────────────────────────────────────────────────────────
# 5. 메인 실행 예시
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # YAL 파일 읽어 모듈 리스트 생성
    yal_file = "./example/ami33.yal"
    modules = parse_yal(yal_file)

    # Chip 객체 생성 후, 초기 배치/시각화
    chip = Chip(modules)
    chip.calculate_coordinates()
    chip.plot_b_tree()

    # 초기 배치 스펙
    print("=== 초기 Chip spec ===")
    w_, h_, area_ = calculate_total_area(chip.modules)
    hpwl_ = calculate_hpwl(chip.modules)
    print(f"BoundingBox area = {area_}, HPWL = {hpwl_}")
    init_area = area_
    init_hpwl = hpwl_
    init_cost = calc_combined_cost(chip.modules, w=0.5)

    # "여러 번 무작위 섞기"
    ans = input("초기 배치를 무작위로 여러 번 섞어보겠습니다. (y/n): ")
    if ans.lower().startswith('y'):
        tries = int(input("몇 번 섞어볼까요?: "))

        old_modules = chip.modules[:]
        old_tree = copy.deepcopy(chip.root)
        old_cost = calc_combined_cost(chip.modules, w=0.5)

        best_random_cost = old_cost
        best_modules_arr = old_modules[:]

        for i in range(tries):
            new_cost = chip.randomize_b_tree(w=0.5)
            print(f"[Random Try {i+1}/{tries}] cost = {new_cost:.3f}")

            # 최소 cost 갱신?
            if new_cost < best_random_cost:
                best_random_cost = new_cost
                best_modules_arr = chip.modules[:]

            # 복원
            chip.modules = old_modules[:]
            chip.root = copy.deepcopy(old_tree)
            chip.calculate_coordinates()

        print(f"[결과] {tries}번 시도 중 최소 cost = {best_random_cost:.3f}")
        ans2 = input("이 배치를 초기배치로 적용할까요? (y/n): ")
        if ans2.lower().startswith('y'):
            chip.modules = best_modules_arr[:]
            chip.build_b_tree()
            chip.calculate_coordinates()
            print(f"초기배치를 cost={best_random_cost:.3f}로 적용했습니다.")
            chip.plot_b_tree()
            plt.show()

    # FastSA 진행 여부
    answer = input("FastSA로 최적화를 진행하시겠습니까? (y/n): ")
    if answer.lower().startswith('y'):
        best_chip = fast_sa(
            chip,
            max_iter=1000,
            P=0.95,
            c=100,
            w=0.5,
            sample_moves=5
        )
        print("=== FastSA 종료 결과 ===")
        final_w, final_h, final_area = calculate_total_area(best_chip.modules)
        final_hpwl = calculate_hpwl(best_chip.modules)
        final_cost = calc_combined_cost(best_chip.modules, w=0.5)
        print(f"초기 BoundingBox area = {init_area}, HPWL = {init_hpwl}")
        print(f"초기 cost = {init_cost:.3f}")
        print(f"BoundingBox area = {final_area}, HPWL = {final_hpwl}")
        print(f"최종 cost = {final_cost:.3f}")

        best_chip.plot_b_tree()
        plt.show()
