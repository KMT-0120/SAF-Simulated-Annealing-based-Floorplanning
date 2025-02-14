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

    def set_position(self, x: float, y: float, order: int):
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

        # Contour: [(x_start, x_end, height), ...]
        self.contour = []

    def build_b_tree(self, root_index=0):
        """
        root_index 에 해당하는 모듈을 트리의 루트로 삼고
        나머지 모듈들을 순서대로 left, right 자식으로 연결 (BFS식).
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

    ################################################
    #   [Contour 기반] B*-Tree 좌표 배치
    ################################################
    def calculate_coordinates(self, node=None, order=1):
        """
        Contour 기반으로 x좌표는
          - 왼쪽 자식(left child)은 parent의 (x + width)에 놓고
          - 오른쪽 자식(right child)은 parent의 x에 놓으며,
        y좌표는 해당 x구간의 contour(최댓 높이) 위에 배치.
        """
        if node is None:
            node = self.root
        if node is None:
            return

        # 루트 노드 배치
        if node.parent is None:
            # 루트는 (0,0) 시작
            x = 0
            y = self.get_contour_height(x, node.module.width)
            node.module.set_position(x, y, order)
            self.update_contour(x, node.module.width, y + node.module.height)
        else:
            # 부모 기준 왼쪽 자식 → x = parent의 x + parent.width
            if node == node.parent.left:
                x = node.parent.module.x + node.parent.module.width
            # 부모 기준 오른쪽 자식 → x = parent의 x
            elif node == node.parent.right:
                x = node.parent.module.x

            y = self.get_contour_height(x, node.module.width)
            node.module.set_position(x, y, order)
            self.update_contour(x, node.module.width, y + node.module.height)

        # 경계 체크 (optional)
        if (node.module.x + node.module.width > self.bound.width or
                node.module.y + node.module.height > self.bound.height):
            raise ValueError(f"Module [{node.module.name}] out of boundary")

        # 재귀적으로 자식 노드 배치
        next_order = order + 1
        if node.left:
            next_order = self.calculate_coordinates(node.left, next_order)
        if node.right:
            next_order = self.calculate_coordinates(node.right, next_order)

        return next_order

    def get_contour_height(self, x_start, width):
        x_end = x_start + width
        max_height = 0.0

        if not self.contour:
            return 0.0

        for (cx_start, cx_end, c_height) in self.contour:
            if cx_end <= x_start or cx_start >= x_end:
                continue
            else:
                max_height = max(max_height, c_height)

        return max_height

    def update_contour(self, x_start, width, new_top):
        x_end = x_start + width
        if not self.contour:
            self.contour.append((x_start, x_end, new_top))
            return

        new_contour = []
        inserted = False

        for (cx_start, cx_end, c_height) in self.contour:
            # 1) 완전히 왼쪽
            if cx_end <= x_start:
                new_contour.append((cx_start, cx_end, c_height))
            # 2) 완전히 오른쪽
            elif cx_start >= x_end:
                if not inserted:
                    new_contour.append((x_start, x_end, new_top))
                    inserted = True
                new_contour.append((cx_start, cx_end, c_height))
            # 3) 부분 겹침
            else:
                # 왼쪽 자투리
                if cx_start < x_start:
                    new_contour.append((cx_start, x_start, c_height))

                # 오른쪽 자투리
                if cx_end > x_end:
                    if not inserted:
                        new_contour.append((x_start, x_end, new_top))
                        inserted = True
                    new_contour.append((x_end, cx_end, c_height))
                else:
                    # (cx_start ~ cx_end)이 전부 새 구간과 겹침
                    if not inserted and cx_end >= x_end:
                        new_contour.append((x_start, x_end, new_top))
                        inserted = True

        if not inserted:
            new_contour.append((x_start, x_end, new_top))

        new_contour.sort(key=lambda seg: seg[0])

        # 인접 구간 merge (optional)
        merged = []
        for seg in new_contour:
            if not merged:
                merged.append(seg)
            else:
                prev = merged[-1]
                if abs(prev[2] - seg[2]) < 1e-9 and abs(prev[1] - seg[0]) < 1e-9:
                    merged[-1] = (prev[0], seg[1], prev[2])
                else:
                    merged.append(seg)

        self.contour = merged

    ################################################
    #   부모–자식 인접(or 위치) 검사 함수들
    ################################################
    def is_left_child_adjacent(self, parent_node: BTreeNode, child_node: BTreeNode) -> bool:
        p = parent_node.module
        c = child_node.module
        horizontally_adjacent = abs((p.x + p.width) - c.x) < 1e-9
        y_overlap = not ( (c.y + c.height) <= p.y or (p.y + p.height) <= c.y )
    
        return horizontally_adjacent and y_overlap

    def is_right_child_adjacent(self, parent_node: BTreeNode, child_node: BTreeNode) -> bool:
        """
        (이전 예시용) 오른쪽 자식은 부모 위에 '접촉'하는지 확인하는 함수였으나,
        지금은 붙어있을 필요가 없어서 사용 안 함.
        """
        p = parent_node.module
        c = child_node.module
        # 기존에는 parent의 top == child의 bottom 등등 확인
        return True  # 지금은 dummy 반환

    def check_all_parent_child_adjacent(self, root: BTreeNode) -> bool:
        """
        - 왼쪽 자식: 반드시 parent의 오른쪽 면 == child의 왼쪽 면
        - 오른쪽 자식: '굳이 붙어있을 필요는 없으며',
          다만 예시로 "부모 위쪽(y + height) 이상에"만 있으면 통과한다고 가정.
          (아무 검사도 안 할거면 아래 로직 자체를 없애도 됨)
        """
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left:
                # 왼쪽 자식 → 부모와 붙었는지 검사
                if not self.is_left_child_adjacent(node, node.left):
                    return False
                stack.append(node.left)
            if node.right:
                # 오른쪽 자식은 단순히 "부모보다 위쪽에 있는지" 정도만 확인
                p = node.module
                c = node.right.module
                # 예시: c.y >= p.y + p.height
                if c.y < p.y + p.height:
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

        ax1.set_title("B*-Tree Physical Placement (Contour-based)")
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
#  부모-자식 인접 (왼쪽만) 배치 찾기
################################################
def try_build_and_place(chip: Chip):
    """
    모든 모듈을 루트로 시도하여 B*-Tree 생성 후,
    calculate_coordinates로 배치.
    그 다음 부모–자식이 전부 "적합한 조건"을 만족하는지 검사.
    만족하면 True를 반환하고 종료,
    끝까지 실패하면 False를 반환.
    """
    for i in range(len(chip.modules)):
        # 1) 트리를 i번째 모듈을 루트로 생성
        chip.build_b_tree(root_index=i)

        # 매번 contour 초기화
        chip.contour = []

        # 2) 배치 실행 (좌표 계산)
        try:
            chip.calculate_coordinates()
        except ValueError:
            # 경계 밖으로 나가면 배치 실패, 다음 루트 시도
            continue

        # 3) 부모–자식 인접(왼쪽만) + 오른쪽 자식은 단순위치 체크
        if chip.root and chip.check_all_parent_child_adjacent(chip.root):
            # 성공 시 True 반환
            print(f"[성공] Root module = {chip.modules[i].name} 일 때, "
                  f"왼쪽 자식 인접 조건 + 오른쪽 자식 위쪽 조건 만족하는 배치 완료!")
            return True

    # 모든 루트 후보를 시도했는데도 실패
    print("non-admissible placement (주어진 부모–자식 조건을 만족하는 배치를 찾지 못함).")
    return False


################################################
#  메인 실행부 (예시)
################################################
if __name__ == "__main__":
    # YAL 파일 경로 (예: ./example/gptexample.yal)
    yal_file = "./example/apte.yal"

    # 1) 모듈들 파싱
    modules = parse_yal(yal_file)

    # 2) Chip 객체 생성
    chip = Chip(modules)

    # 3) 부모–자식 조건(왼쪽만 접촉, 오른쪽은 위에) 만족 배치 찾기
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
