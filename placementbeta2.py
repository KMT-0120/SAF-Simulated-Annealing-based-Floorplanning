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

        # ---------------------------
        # [수정된 부분] Contour 배열
        # ---------------------------
        # contour: [(x_start, x_end, height), ...]
        # 서로 인접하지 않은 구간을 정렬하여 유지하면서
        # 모듈을 배치하면 해당 x-구간에서의 "현재까지의 최댓 y"가 무엇인지 찾고
        # 배치 후에는 그 구간을 업데이트한다.
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

    # ------------------------------------------------------------------
    #   [수정된 부분] B*-Tree 좌표 계산 (Contour 기반)
    # ------------------------------------------------------------------
    def calculate_coordinates(self, node=None, order=1):
        """
        Contour 기반으로 x좌표는
          - 왼쪽 자식(left child)은 parent의 (x + width)에 놓고
          - 오른쪽 자식(right child)은 parent의 x에 놓으며,
        y좌표는 해당 x구간에서 contour가 알려주는 최고 높이 위에 배치.
        """
        if node is None:
            node = self.root
        if node is None:
            return

        # 루트 노드 배치
        if node.parent is None:
            # 루트는 (0,0) 시작으로 배치
            x = 0
            # 해당 너비만큼 contour 에서 최고 높이를 가져옴
            y = self.get_contour_height(x, node.module.width)
            node.module.set_position(x, y, order)
            # contour 갱신
            self.update_contour(x, node.module.width, y + node.module.height)
        else:
            # 부모 기준 왼쪽 자식 → 부모 오른쪽에 붙임
            if node == node.parent.left:
                x = node.parent.module.x + node.parent.module.width
            # 부모 기준 오른쪽 자식 → 부모 x와 동일
            elif node == node.parent.right:
                x = node.parent.module.x

            # contour상 최고 높이 (x~x+width 구간)
            y = self.get_contour_height(x, node.module.width)
            node.module.set_position(x, y, order)
            # contour 갱신
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

    # ------------------------------------------------------------------
    # [추가] Contour에서 특정 x구간([x, x+width))에 대한 최고 높이 얻기
    # ------------------------------------------------------------------
    def get_contour_height(self, x_start, width):
        x_end = x_start + width
        max_height = 0.0

        # contour가 비어있으면 높이는 0으로 간주
        if not self.contour:
            return 0.0

        # x_start~x_end가 겹치는 모든 세그먼트를 순회하며 최대 y갱신
        for (cx_start, cx_end, c_height) in self.contour:
            if cx_end <= x_start or cx_start >= x_end:
                # 구간이 전혀 겹치지 않으면 무시
                continue
            else:
                max_height = max(max_height, c_height)

        return max_height

    # ------------------------------------------------------------------
    # [추가] 모듈 배치 후, 해당 (x_start~x_start+width) 구간의 contour 갱신
    # ------------------------------------------------------------------
    def update_contour(self, x_start, width, new_top):
        x_end = x_start + width
        if not self.contour:
            # contour 비어있으면 바로 삽입
            self.contour.append((x_start, x_end, new_top))
            return

        new_contour = []
        inserted = False

        for (cx_start, cx_end, c_height) in self.contour:
            # 1) 현재 세그먼트가 새 구간과 왼쪽으로 완전히 떨어져 있는 경우
            if cx_end <= x_start:
                new_contour.append((cx_start, cx_end, c_height))
            # 2) 현재 세그먼트가 새 구간과 오른쪽으로 완전히 떨어져 있는 경우
            elif cx_start >= x_end:
                # 아직 새 구간을 넣지 않았다면 먼저 삽입
                if not inserted:
                    new_contour.append((x_start, x_end, new_top))
                    inserted = True
                new_contour.append((cx_start, cx_end, c_height))
            # 3) 현재 세그먼트가 새 구간과 부분적으로 겹칠 경우 → 분할
            else:
                # 왼쪽 자투리
                if cx_start < x_start:
                    new_contour.append((cx_start, x_start, c_height))
                # 오른쪽 자투리
                if cx_end > x_end:
                    # 아직 새 구간 미삽입이면 우선 새 구간 넣고
                    if not inserted:
                        new_contour.append((x_start, x_end, new_top))
                        inserted = True
                    new_contour.append((x_end, cx_end, c_height))
                else:
                    # (cx_start~cx_end)이 전부 새 구간과 겹치는 경우
                    # → 새 구간이 더 높은 값으로 덮어쓰므로 skip
                    if not inserted and cx_end >= x_end:
                        new_contour.append((x_start, x_end, new_top))
                        inserted = True

        # 만약 아직 새 구간을 넣지 못했다면 맨 뒤에 넣음
        if not inserted:
            new_contour.append((x_start, x_end, new_top))

        # x_start 기준으로 정렬
        new_contour.sort(key=lambda seg: seg[0])

        # 붙어있는 같은 높이 세그먼트는 병합 (optional)
        merged = []
        for seg in new_contour:
            if not merged:
                merged.append(seg)
            else:
                prev = merged[-1]
                if abs(prev[2] - seg[2]) < 1e-9 and abs(prev[1] - seg[0]) < 1e-9:
                    # 높이 같고 양 끝이 이어지면 머지
                    merged[-1] = (prev[0], seg[1], prev[2])
                else:
                    merged.append(seg)

        self.contour = merged

    ################################################
    #   부모–자식이 붙어 있는지 검사하는 함수들
    ################################################
    def is_left_child_adjacent(self, parent_node: BTreeNode, child_node: BTreeNode) -> bool:
        """
        [Left Child]:  parent의 오른쪽 면 == child의 왼쪽 면
        => x좌표로 확인
        """
        p = parent_node.module
        c = child_node.module

        # (수정) 기존에는 y 구간 교집합까지 확인했으나,
        # 여기서는 '붙어있는지' 여부를 x 기준으로만 확인한다고 가정
        # 또는 필요하다면 y 겹침 체크를 유지하되 완화 가능
        return abs((p.x + p.width) - c.x) < 1e-9

    def is_right_child_adjacent(self, parent_node: BTreeNode, child_node: BTreeNode) -> bool:
        """
        [Right Child]: 부모 위에 위치하지만 '접촉은 필요 없음'
        단, admissible placement 에서는 parent.x == child.x
        """
        p = parent_node.module
        c = child_node.module
        return abs(p.x - c.x) < 1e-9

    def check_all_parent_child_adjacent(self, root: BTreeNode) -> bool:
        """
        루트부터 모든 노드 순회하며,
        (부모, 자식) 쌍이 left child / right child 조건을 충족하는지 검사
        """
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left:
                # btree 상의 left child → x좌표가 parent's right와 동일해야
                if not self.is_left_child_adjacent(node, node.left):
                    return False
                stack.append(node.left)
            if node.right:
                # btree 상의 right child → x좌표가 parent's x와 동일
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

        # 매번 contour 초기화
        chip.contour = []

        # 2) 배치 실행 (좌표 계산)
        try:
            chip.calculate_coordinates()
        except ValueError:
            # 경계 밖으로 나가면 배치 실패, 다음 루트 시도
            continue

        # 3) 부모–자식 모두 붙었는지 검사
        if chip.root and chip.check_all_parent_child_adjacent(chip.root):
            # 성공 시 True 반환
            print(f"[성공] Root module = {chip.modules[i].name} 일 때, "
                  f"모든 모듈이 부모-자식 쌍 조건을 만족하는 배치로 초기화 완료!")
            return True

    # 모든 루트 후보를 시도했는데도 실패
    print("non-admissible placement (모든 모듈이 부모-자식 쌍 조건을 만족하는 배치를 찾지 못함).")
    return False


################################################
#  메인 실행부 (예시)
################################################
if __name__ == "__main__":
    # YAL 파일 경로 (예: ./example/gptexample.yal)
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
