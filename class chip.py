class Chip:
    def __init__(self):
        self.modules = []  # Module 리스트
        self.arrangement = None  # B*-Tree 구조
        self.cost = {"hpwl": 0, "area": 0}  # 비용 정보

    def add_module(self, module):
        self.modules.append(module)

    def apply_b_star_tree(self):
        """B*-Tree를 사용하여 배치를 계산"""
        pass

    def evaluate_cost(self):
        """HPWL 및 총 면적 계산"""
        self.cost["hpwl"] = self.calculate_hpwl()
        self.cost["area"] = self.calculate_total_area()

    def calculate_hpwl(self):
        """배선 길이(HPWL)를 계산"""
        total_hpwl = 0
        for module in self.modules:
            if module.net:  # 연결된 네트워크가 있는 경우
                x_coords = [module.x]
                y_coords = [module.y]
                for connected_name in module.net:
                    connected_module = next((m for m in self.modules if m.name == connected_name), None)
                    if connected_module:
                        x_coords.append(connected_module.x)
                        y_coords.append(connected_module.y)
                hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
                total_hpwl += hpwl
        return total_hpwl

    def calculate_total_area(self):
        """총 면적 계산"""
        return sum(module.area for module in self.modules)

    def __str__(self):
        return (f"Chip with {len(self.modules)} modules, "
                f"Cost: {self.cost}")