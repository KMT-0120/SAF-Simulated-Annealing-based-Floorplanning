import matplotlib.pyplot as plt
import networkx as nx

#####################
#      Module       #
#####################
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
        self.order = None  # 순서 번호 필드 추가

    def set_position(self, x: float, y: float, order: int):  # 위치 설정 시 순서 번호도 설정
        self.x = x
        self.y = y
        self.order = order

    def __str__(self):
        formatted_net = "\n    ".join(self.net)
        return (f"Module(Name={self.name}, Width={self.width:.2f}, Height={self.height:.2f}, "
                f"Area={self.area:.2f}, Position=({self.x:.2f}, {self.y:.2f}), "
                f"Order={self.order}, Type={self.type}, Net=\n    {formatted_net})")


#####################
#     BTreeNode     #
#####################
class BTreeNode:
    def __init__(self, module, parent=None):
        self.module = module
        self.left = None
        self.right = None
        self.parent = parent


#####################
#        Chip       #
#####################
class Chip:
    def __init__(self, modules, default_width=1000, default_height=1000):
        # 'PARENT' 모듈이 아닌 것만 실제 배치 대상 모듈로 사용
        self.modules = [m for m in modules if m.type != 'PARENT']
        # 'PARENT' 타입 모듈(칩 경계)을 찾음
        self.bound = next((m for m in modules if m.type == 'PARENT'), None)

        # 기본 'PARENT' 모듈 설정 (없을 경우)
        if not self.bound:
            self.bound = Module(name='DefaultParent',
                                width=default_width,
                                height=default_height,
                                module_type='PARENT')
            print("No 'PARENT' module found. Using default dimensions.")

        self.root = None  # B*-Tree 루트 노드

    def build_b_tree(self, root_index=0):
        """
        root_index 에 해당하는 모듈을 트리의 루트로 삼고
        나머지 모듈들을 순서대로 left, right 자식으로 연결.
        """
        if not self.modules or not self.bound:
            return

        # 루트 설정
        chosen_root = self.modules[root_index]
        self.root = BTreeNode(chosen_root)
        queue = [self.root]

        # 루트를 제외한 나머지 모듈들
        remaining_modules = [m for i, m in enumerate(self.modules) if i != root_index]

        # BFS 방식으로 왼자식, 오른자식 배정
        for module in remaining_modules:
            parent = queue[0]
            if not parent.left:
                parent.left = BTreeNode(module, parent)
                queue.append(parent.left)
            elif not parent.right:
                parent.right = BTreeNode(module, parent)
                queue.append(parent.right)
                # 왼자식, 오른자식 둘 다 찼으니 queue에서 pop
                queue.pop(0)

    # B*-Tree를 기반으로 좌표를 할당하면서, 배치 순서를 업데이트
    def calculate_coordinates(self, node=None, x_offset=0, y_offset=0,
                              placed_modules=None, order=1):
        if node is None:
            node = self.root
        if node is None:
            return

        if placed_modules is None:
            placed_modules = []

        # 루트 노드인 경우
        if not node.parent:
            x, y = x_offset, y_offset
        else:
            # 부모 기준으로 왼자식은 부모의 오른쪽에 배치
            if node == node.parent.left:
                x = node.parent.module.x + node.parent.module.width
                y = node.parent.module.y
            # 부모 기준으로 오른자식은 부모의 위쪽에 배치
            elif node == node.parent.right:
                x = node.parent.module.x
                y = node.parent.module.y + node.parent.module.height

        # 겹치거나 경계를 벗어나지 않도록 위치를 조정
        x, y = self.adjust_for_overlaps(node.module, x, y, placed_modules)

        # 최종 위치, 순서 번호 설정
        node.module.set_position(x, y, order)
        placed_modules.append(node.module)

        if node.left:
            order = self.calculate_coordinates(node.left, x, y, placed_modules, order + 1)
        if node.right:
            order = self.calculate_coordinates(node.right, x, y, placed_modules, order + 1)
        return order

    def adjust_for_overlaps(self, module, x_start, y_start, placed_modules):
        """
        (x_start, y_start)에 모듈을 놓았을 때
        1) 경계 내에 들어가고
        2) 이미 놓인 모듈과 겹치지 않는
        위치를 찾아서 반환
        """
        x, y = x_start, y_start
        while True:
            if self.is_within_bounds(module, x, y) and not self.check_overlap(module, x, y, placed_modules):
                return x, y
            # 조금씩 x를 오른쪽으로 이동하며 겹침 회피
            x += 1
            # 경계 넘어가면 y를 1 증가시키고 x를 0으로 리셋
            if x + module.width > self.bound.width:
                x = 0
                y += 1
            if y + module.height > self.bound.height:
                raise ValueError("Unable to fit module within bounds")

    def check_overlap(self, module, x, y, placed_modules):
        """
        (x, y)에 모듈을 배치했을 때 기존 placed_modules와 겹치는지 확인
        """
        for placed in placed_modules:
            # 직사각형 간 overlap 검사
            if (x < placed.x + placed.width and
                x + module.width > placed.x and
                y < placed.y + placed.height and
                y + module.height > placed.y):
                return True
        return False

    def is_within_bounds(self, module, x, y):
        """
        경계 모듈(bound) 범위 안에 들어가는지 확인
        """
        return (x + module.width <= self.bound.width and
                y + module.height <= self.bound.height)

    ################################################
    #   부모–자식이 붙어 있는지 검사하는 함수들
    ################################################
    def is_left_child_adjacent(self, parent_node: BTreeNode, child_node: BTreeNode) -> bool:
        """
        [Left Child] → 부모의 오른쪽 면 == 자식의 왼쪽 면,
        그리고 (parent.y 범위)와 (child.y 범위)가 교집합이 있어야 함.
        """
        p = parent_node.module
        c = child_node.module

        p_left, p_right = p.x, p.x + p.width
        p_bottom, p_top = p.y, p.y + p.height

        c_left, c_right = c.x, c.x + c.width
        c_bottom, c_top = c.y, c.y + c.height

        # 1) 수평으로 정확히 맞닿아 있는지?
        #    즉 부모의 오른쪽 == 자식의 왼쪽
        if abs(p_right - c_left) > 1e-9:
            return False

        # 2) 수직 구간이 교집합을 갖는지? (y-interval overlap)
        #    교집합이 없으면 (p_top <= c_bottom) or (c_top <= p_bottom)
        if p_top <= c_bottom or c_top <= p_bottom:
            return False

        return True

    def is_right_child_adjacent(self, parent_node: BTreeNode, child_node: BTreeNode) -> bool:
        """
        [Right Child] → 부모의 위쪽 면 == 자식의 아래쪽 면,
        그리고 (parent.x 범위)와 (child.x 범위)가 교집합이 있어야 함.
        """
        p = parent_node.module
        c = child_node.module

        p_left, p_right = p.x, p.x + p.width
        p_bottom, p_top = p.y, p.y + p.height

        c_left, c_right = c.x, c.x + c.width
        c_bottom, c_top = c.y, c.y + c.height

        # 1) 수직으로 정확히 맞닿아 있는지?
        #    즉 부모의 위쪽 == 자식의 아래쪽
        if abs(p_top - c_bottom) > 1e-9:
            return False

        # 2) 수평 구간이 교집합을 갖는지? (x-interval overlap)
        #    교집합이 없으면 (p_right <= c_left) or (c_right <= p_left)
        if p_right <= c_left or c_right <= p_left:
            return False

        return True

    def check_all_parent_child_adjacent(self, root: BTreeNode) -> bool:
        """
        루트부터 모든 노드 순회하며, (부모, 자식) 쌍이
        - left child 일 때는 is_left_child_adjacent
        - right child 일 때는 is_right_child_adjacent
        를 만족하는지 검사
        """
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left:
                # btree 상의 left child → '오른쪽 면' 맞닿고 y-range 겹침
                if not self.is_left_child_adjacent(node, node.left):
                    return False
                stack.append(node.left)
            if node.right:
                # btree 상의 right child → '위쪽 면' 맞닿고 x-range 겹침
                if not self.is_right_child_adjacent(node, node.right):
                    return False
                stack.append(node.right)
        return True

    ################################################
    # 시각화 함수들
    ################################################
    def plot_b_tree(self):
        """
        B*-Tree 배치 후 물리좌표 상에서 모듈들의 위치와 경계 모듈을 시각화
        """
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111)
        self._plot_node(self.root, ax1)

        # 경계 모듈(Parent) 그리기
        bound_rect = plt.Rectangle((0, 0), self.bound.width, self.bound.height,
                                   edgecolor='red', facecolor='none', lw=2)
        ax1.add_patch(bound_rect)
        ax1.text(self.bound.width / 2, self.bound.height / 2,
                 self.bound.name, ha='center', va='center',
                 fontsize=10, color='red')

        ax1.set_title("B*-Tree Physical Placement")
        ax1.set_xlabel("X-coordinate")
        ax1.set_ylabel("Y-coordinate")
        ax1.set_xlim(0, self.bound.width + 50)
        ax1.set_ylim(0, self.bound.height + 50)

    def _plot_node(self, node, ax):
        """
        재귀적으로 BTreeNode를 순회하면서 직사각형(모듈)을 그려준다.
        """
        if node is None:
            return
        module = node.module
        rect = plt.Rectangle((module.x, module.y),
                             module.width, module.height,
                             edgecolor='blue',
                             facecolor='lightblue',
                             fill=True, lw=2)
        ax.add_patch(rect)
        ax.text(module.x + module.width / 2,
                module.y + module.height / 2,
                f'{module.name}\n#{module.order}',
                ha='center', va='center', fontsize=8)
        self._plot_node(node.left, ax)
        self._plot_node(node.right, ax)

    def plot_b_tree_structure(self):
        """
        B*-Tree 구조 (그래프) 자체를 네트워크 그래프로 시각화
        """
        if self.root is None:
            return
        G = nx.DiGraph()
        pos = {}

        def add_edges(node, parent_name=None, depth=0, x_pos=0):
            if node is None:
                return x_pos
            node_name = node.module.name
            pos[node_name] = (x_pos, -depth)
            G.add_node(node_name, label=node.module.name)
            if parent_name:
                G.add_edge(parent_name, node_name)
            x_pos = add_edges(node.left, node_name, depth + 1, x_pos)
            x_pos = add_edges(node.right, node_name, depth + 1, x_pos + 1)
            return x_pos

        add_edges(self.root)
        fig2 = plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True,
                node_color="lightblue", edge_color="gray",
                node_size=2000, font_size=10)
        plt.title("B*-Tree Structure")


################################################
#     유틸리티 함수들 (HPWL, total area 등)
################################################
def calculate_hpwl(modules):
    """
    HPWL: Half-Perimeter Wire Length
    (여기서는 간단히 각 모듈 bounding box 둘레의 절반)
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
    배치된 모듈들을 모두 감싸는 최소 bounding box의 면적을 반환
    """
    if not modules:
        return 0, 0, 0
    min_x = min(m.x for m in modules)
    max_x = max(m.x + m.width for m in modules)
    min_y = min(m.y for m in modules)
    max_y = max(m.y + m.height for m in modules)

    chip_width = max_x - min_x
    chip_height = max_y - min_y
    total_area = chip_width * chip_height
    return chip_width, chip_height, total_area


def parse_yal(file_path):
    """
    YAL 형식 파일 파싱
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
                                                 .replace(";", "")
                                                 .split()))
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


################################################
#  부모-자식 인접 조건을 만족하는 배치 찾기
################################################
def try_build_and_place(chip: Chip):
    """
    모든 모듈을 루트로 시도하여 B*-Tree 생성 후,
    calculate_coordinates로 배치.
    그 다음 부모–자식이 전부 "붙어있는지" 검사.
    붙어있으면 True를 반환하고 종료,
    끝까지 실패하면 False를 반환.
    """
    for i in range(len(chip.modules)):
        # 1) 트리를 i번째 모듈을 루트로 생성
        chip.build_b_tree(root_index=i)

        # 2) 배치 실행 (좌표 초기화)
        try:
            chip.calculate_coordinates()
        except ValueError:
            # 경계 밖으로 나가면 배치 실패, 다음 루트 시도
            continue

        # 3) 부모–자식 모두 붙었는지 검사
        if chip.root and chip.check_all_parent_child_adjacent(chip.root):
            # 성공 시 True 반환
            print(f"[성공] Root module = {chip.modules[i].name} 일 때, 모든 모듈이 부모-자식 쌍이 붙는 배치로 b*-tree를 초기화 할 수 있음 [admissible placement]")
            return True

    # 모든 루트 후보를 시도했는데도 실패
    print("non-admissible placement (모든 모듈이 부모-자식 쌍이 붙는 배치로 b*-tree를 초기화 할 수 없음).")
    return False


################################################
#  메인 실행부 (예시)
################################################
if __name__ == "__main__":
    # YAL 파일 경로
    yal_file = "./example/gptexample.yal"

    # 1) 모듈들 파싱
    modules = parse_yal(yal_file)

    # 2) Chip 객체 생성
    chip = Chip(modules)

    # 3) 부모-자식 붙어있음 조건을 만족하는 배치 찾기
    success = try_build_and_place(chip)

    if success:
        # 성공한 배치가 있으면 시각화
        chip.plot_b_tree()
        chip.plot_b_tree_structure()
        plt.show()

        # 면적, HPWL 출력
        chip_width, chip_height, total_area = calculate_total_area(chip.modules)
        print(f"칩의 전체 영역: 가로={chip_width}, 세로={chip_height}, 면적={total_area}")
        print("HPWL:", calculate_hpwl(chip.modules))
    else:
        # 실패 메시지는 이미 try_build_and_place 안에서 출력됨
        pass
