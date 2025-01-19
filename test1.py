import matplotlib.pyplot as plt

class Module:

    def __init__(self, name: str, width: float, height: float, module_type: str, net=None):

        self.name = name
        self.width = width
        self.height = height
        self.area = width * height
        self.x = 0
        self.y = 0
        self.type = module_type  # 문자열로 저장
        self.net = net if net is not None else []

    def set_position(self, x: float, y: float):

        self.x = x
        self.y = y

    def __str__(self):
        # Net 데이터를 깔끔하게 정리해서 출력
        formatted_net = "\n    ".join(self.net)

        return (f"Module(Name={self.name}, Width={self.width:.2f}, Height={self.height:.2f}, "
                f"Area={self.area:.2f}, Position=({self.x:.2f}, {self.y:.2f}), "
                f"Type={self.type}, Net=\n    {formatted_net})")

def parse_yal(file_path):
    """
    YAL 파일을 파싱하여 Module 객체 리스트로 반환
    """
    modules = []  # Module 객체 리스트
    with open(file_path, 'r') as f:
        lines = f.readlines()

    module_data = {}
    in_network = False

    for line in lines:
        line = line.strip()

        # 빈 줄 및 주석 무시
        if not line or line.startswith("/*") or line.startswith("*"):
            continue

        # MODULE 시작
        if line.startswith("MODULE"):
            module_name = line.split()[1].strip(';')
            module_data = {'name': module_name, 'net': []}
            continue

        # TYPE 읽기
        if line.startswith("TYPE"):
            module_type = line.split()[1].strip(';')  # 문자열 그대로 저장
            module_data['type'] = module_type
            continue

        # DIMENSIONS 읽기
        if line.startswith("DIMENSIONS"):
            dimensions = list(map(float, line.replace("DIMENSIONS", "").replace(";", "").split()))
            if len(dimensions) >= 8:
                # 좌표로부터 너비와 높이 계산
                x_coords = [dimensions[i] for i in range(0, len(dimensions), 2)]
                y_coords = [dimensions[i + 1] for i in range(0, len(dimensions), 2)]
                module_data['width'] = max(x_coords) - min(x_coords)
                module_data['height'] = max(y_coords) - min(y_coords)
            else:
                raise ValueError("DIMENSIONS must contain at least 8 values (four points).")
            continue

        # NETWORK 시작
        if line.startswith("NETWORK"):
            in_network = True
            continue

        # ENDNETWORK
        if line.startswith("ENDNETWORK"):
            in_network = False
            continue

        # ENDMODULE: Module 저장
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

        # NETWORK 데이터 수집
        if in_network:
            module_data['net'].append(line.strip(';'))
            continue

    return modules

class BTreeNode:
    def __init__(self, module):
        self.module = module
        self.left = None
        self.right = None

class Chip:
    def __init__(self, modules):
        self.modules = modules
        self.root = None

    def build_b_tree(self):
        """모듈 리스트를 B*-Tree 형태로 정렬"""
        if not self.modules:
            return

        self.root = BTreeNode(self.modules[0])
        queue = [self.root]

        for module in self.modules[1:]:
            parent = queue[0]  # 부모 노드 선택

            if not parent.left:
                parent.left = BTreeNode(module)
                queue.append(parent.left)
            elif not parent.right:
                parent.right = BTreeNode(module)
                queue.append(parent.right)
                queue.pop(0)  # 부모가 왼쪽과 오른쪽 자식을 모두 가졌으면 큐에서 제거

    def calculate_coordinates(self, node=None, x_offset=0, y_offset=0):
        """B*-Tree 노드의 물리적 위치 계산"""
        if node is None:
            node = self.root

        if node is None:
            return

        # 현재 노드 배치
        node.module.set_position(x_offset, y_offset)

        # 왼쪽 자식은 오른쪽으로 이동
        if node.left:
            self.calculate_coordinates(node.left, x_offset + node.module.width, y_offset)

        # 오른쪽 자식은 위로 이동
        if node.right:
            self.calculate_coordinates(node.right, x_offset, y_offset + node.module.height)

    def display_b_tree(self, node=None, level=0):
        """B*-Tree를 텍스트로 출력"""
        if node is None:
            node = self.root

        if node is not None:
            print("  " * level + f"{node.module.name}: ({node.module.x}, {node.module.y})")
            if node.left:
                self.display_b_tree(node.left, level + 1)
            if node.right:
                self.display_b_tree(node.right, level + 1)

    def plot_b_tree(self):
        """B*-Tree 배치를 플롯으로 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_node(self.root, ax)

        ax.set_title("B*-Tree Physical Placement")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.axis("equal")
        plt.show()

    def _plot_node(self, node, ax):
        """재귀적으로 노드를 플롯"""
        if node is None:
            return

        module = node.module
        rect = plt.Rectangle((module.x, module.y), module.width, module.height,
                             edgecolor='blue', facecolor='lightblue', fill=True, lw=2)
        ax.add_patch(rect)
        ax.text(module.x + module.width / 2, module.y + module.height / 2,
                module.name, ha='center', va='center', fontsize=8)

        # 자식 노드로 이동
        self._plot_node(node.left, ax)
        self._plot_node(node.right, ax)

# 파일 경로 설정
yal_file = "./example/ami33.yal"  # YAL 파일 경로

# YAL 파일 파싱
modules = parse_yal(yal_file)

# Chip 생성 및 B*-Tree 구축
chip = Chip(modules)
chip.build_b_tree()
chip.calculate_coordinates()

# 결과 출력
chip.display_b_tree()

# 배치 결과 플롯
chip.plot_b_tree()
