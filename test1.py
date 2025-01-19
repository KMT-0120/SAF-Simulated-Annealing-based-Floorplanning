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

    def set_position(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        formatted_net = "\n    ".join(self.net)
        return (f"Module(Name={self.name}, Width={self.width:.2f}, Height={self.height:.2f}, "
                f"Area={self.area:.2f}, Position=({self.x:.2f}, {self.y:.2f}), "
                f"Type={self.type}, Net=\n    {formatted_net})")

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

class BTreeNode:
    def __init__(self, module):
        self.module = module
        self.left = None
        self.right = None

class Chip:
    def __init__(self, modules):
        self.modules = [m for m in modules if m.type != 'PARENT']
        self.bound = next((m for m in modules if m.type == 'PARENT'), None)
        self.root = None

    def build_b_tree(self):
        if not self.modules or not self.bound:
            return
        self.root = BTreeNode(self.modules[0])
        queue = [self.root]

        for module in self.modules[1:]:
            parent = queue[0]
            if not parent.left:
                parent.left = BTreeNode(module)
                queue.append(parent.left)
            elif not parent.right:
                parent.right = BTreeNode(module)
                queue.append(parent.right)
                queue.pop(0)

    def calculate_coordinates(self, node=None, x_offset=0, y_offset=0, placed_modules=None):
        if node is None:
            node = self.root
        if node is None:
            return

        if placed_modules is None:
            placed_modules = []

        x, y = self.find_non_overlapping_position(node.module, x_offset, y_offset, placed_modules)

        if x + node.module.width > self.bound.width:
            x = 0
            y_offset += max(m.height for m in placed_modules) if placed_modules else 0
            x, y = self.find_non_overlapping_position(node.module, x, y_offset, placed_modules)

        if y + node.module.height > self.bound.height:
            raise ValueError(f"Module {node.module.name} exceeds boundary constraints")

        node.module.set_position(x, y)
        placed_modules.append(node.module)

        if node.left:
            self.calculate_coordinates(node.left, x + node.module.width, y, placed_modules)

        if node.right:
            self.calculate_coordinates(node.right, x, y + node.module.height, placed_modules)

    def find_non_overlapping_position(self, module, x_start, y_start, placed_modules):
        x, y = x_start, y_start
        while True:
            overlap = False
            for placed in placed_modules:
                if (x < placed.x + placed.width and
                    x + module.width > placed.x and
                    y < placed.y + placed.height and
                    y + module.height > placed.y):
                    overlap = True
                    x += placed.width
                    if x + module.width > self.bound.width:
                        x = 0
                        y += placed.height
                    break
            if not overlap:
                return x, y

    def plot_b_tree(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_node(self.root, ax)

        bound_rect = plt.Rectangle((0, 0), self.bound.width, self.bound.height, 
                                   edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(bound_rect)
        ax.text(self.bound.width / 2, self.bound.height / 2, self.bound.name,
                ha='center', va='center', fontsize=10, color='red')

        ax.set_title("B*-Tree Physical Placement")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.set_xlim(0, self.bound.width + 100)
        ax.set_ylim(0, self.bound.height + 100)
        plt.show()

    def _plot_node(self, node, ax):
        if node is None:
            return
        module = node.module
        rect = plt.Rectangle((module.x, module.y), module.width, module.height,
                             edgecolor='blue', facecolor='lightblue', fill=True, lw=2)
        ax.add_patch(rect)
        ax.text(module.x + module.width / 2, module.y + module.height / 2,
                module.name, ha='center', va='center', fontsize=8)
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
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
        plt.title("B*-Tree Structure")
        plt.show()

# 실행
yal_file = "./example/ami33.yal"
modules = parse_yal(yal_file)
chip = Chip(modules)
chip.build_b_tree()
chip.calculate_coordinates()
chip.plot_b_tree()
chip.plot_b_tree_structure()
