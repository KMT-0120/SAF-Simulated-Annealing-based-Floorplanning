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


# 파일 경로 설정
yal_file = "./example/ami33.yal"  # YAL 파일 경로

# YAL 파일 파싱
modules = parse_yal(yal_file)

# 결과 출력
for module in modules:
    print(module)
