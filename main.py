import matplotlib.pyplot as plt
import networkx as nx

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

        # 기본 'PARENT' 모듈 설정
        if not self.bound:
            self.bound = Module(name='DefaultParent', width=default_width, height=default_height, module_type='PARENT')
            print("No 'PARENT' module found. Using default dimensions.")

    def build_b_tree(self):
        if not self.modules or not self.bound:
            return
        self.root = BTreeNode(self.modules[0])
        queue = [self.root]

        for module in self.modules[1:]:
            parent = queue[0]
            if not parent.left:
                parent.left = BTreeNode(module, parent)
                queue.append(parent.left)
            elif not parent.right:
                parent.right = BTreeNode(module, parent)
                queue.append(parent.right)
                queue.pop(0)

    # calculate_coordinates 메서드에서 모듈을 배치할 때마다 순서를 업데이트
    def calculate_coordinates(self, node=None, x_offset=0, y_offset=0, placed_modules=None, order=1):
        if node is None:
            node = self.root
        if node is None:
            return

        if placed_modules is None:
            placed_modules = []

        if not node.parent:  # Root node case
            x, y = x_offset, y_offset
        else:
            # Placement for the left child to the right of the parent
            if node == node.parent.left:
                x = node.parent.module.x + node.parent.module.width
                y = node.parent.module.y
            # Placement for the right child right above the parent
            elif node == node.parent.right:
                x = node.parent.module.x
                y = node.parent.module.y + node.parent.module.height

        # Overlap prevention and boundary check logic
        x, y = self.adjust_for_overlaps(node.module, x, y, placed_modules)

        node.module.set_position(x, y, order)  # 배치 순서 업데이트
        placed_modules.append(node.module)

        if node.left:
            order = self.calculate_coordinates(node.left, x, y, placed_modules, order + 1)
        if node.right:
            order = self.calculate_coordinates(node.right, x, y, placed_modules, order + 1)
        return order

    def adjust_for_overlaps(self, module, x_start, y_start, placed_modules):
        x, y = x_start, y_start
        while True:
            if self.is_within_bounds(module, x, y) and not self.check_overlap(module, x, y, placed_modules):
                return x, y
            # Adjust position incrementally to avoid overlaps
            x += 1
            if x + module.width > self.bound.width:  # Adjust y if x exceeds bounds
                x = 0
                y += 1
            if y + module.height > self.bound.height:
                raise ValueError("Unable to fit module within bounds")

    def check_overlap(self, module, x, y, placed_modules):
        for placed in placed_modules:
            if (x < placed.x + placed.width and x + module.width > placed.x and
                y < placed.y + placed.height and y + module.height > placed.y):
                return True
        return False

    def is_within_bounds(self, module, x, y):
        return x + module.width <= self.bound.width and y + module.height <= self.bound.height

    def plot_b_tree(self):
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111)
        self._plot_node(self.root, ax1)

        bound_rect = plt.Rectangle((0, 0), self.bound.width, self.bound.height, 
                                edgecolor='red', facecolor='none', lw=2)
        ax1.add_patch(bound_rect)
        ax1.text(self.bound.width / 2, self.bound.height / 2, self.bound.name,
                ha='center', va='center', fontsize=10, color='red')

        ax1.set_title("B*-Tree Physical Placement")
        ax1.set_xlabel("X-coordinate")
        ax1.set_ylabel("Y-coordinate")
        ax1.set_xlim(0, self.bound.width + 100)
        ax1.set_ylim(0, self.bound.height + 100)

    # _plot_node 메서드에서 순서 번호를 표시
    def _plot_node(self, node, ax):
        if node is None:
            return
        module = node.module
        rect = plt.Rectangle((module.x, module.y), module.width, module.height,
                            edgecolor='blue', facecolor='lightblue', fill=True, lw=2)
        ax.add_patch(rect)
        ax.text(module.x + module.width / 2, module.y + module.height / 2,
                f'{module.name}\n#{module.order}', ha='center', va='center', fontsize=8)  # 모듈 이름과 순서 번호 표시
        self._plot_node(node.left, ax)
        self._plot_node(node.right, ax)

    def plot_b_tree_structure(self):
        """B*-Tree 구조를 시각화"""
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
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
        plt.title("B*-Tree Structure")

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

# 실행
yal_file = "./example/standardcell.yal"
modules = parse_yal(yal_file)
chip = Chip(modules)
chip.build_b_tree()
chip.calculate_coordinates()
chip.plot_b_tree()
chip.plot_b_tree_structure()
plt.show()
