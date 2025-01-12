class Module:
    def __init__(self, name, module_type, dimensions, iolist, network):
        self.name = name                # MODULE 이름
        self.module_type = module_type  # TYPE (e.g., GENERAL)
        self.dimensions = dimensions    # DIMENSIONS 데이터 (리스트 형태)
        self.iolist = iolist            # IOLIST (리스트 형태)
        self.network = network          # NETWORK 데이터 (리스트 형태)

    def __repr__(self):
        return (f"Module(name={self.name}, type={self.module_type}, "
                f"dimensions={self.dimensions}, iolist={len(self.iolist)} IOs, "
                f"network={len(self.network)} nets)")


def parse_yal(file_path):
    """
    YAL 파일을 파싱하여 Module 객체 리스트로 반환
    """
    modules = []  # Module 객체 리스트
    with open(file_path, 'r') as f:
        lines = f.readlines()

    module_data = {}
    in_iolist = False
    in_network = False

    for line in lines:
        line = line.strip()
        
        # 빈 줄 및 주석 무시
        if not line or line.startswith("/*") or line.startswith("*"):
            continue
        
        # MODULE 시작
        if line.startswith("MODULE"):
            module_name = line.split()[1].strip(';')
            module_data = {'name': module_name, 'iolist': [], 'network': []}
            continue
        
        # TYPE 읽기
        if line.startswith("TYPE"):
            module_data['type'] = line.split()[1].strip(';')
            continue
        
        # DIMENSIONS 읽기
        if line.startswith("DIMENSIONS"):
            dimensions = list(map(int, line.replace("DIMENSIONS", "").replace(";", "").split()))
            module_data['dimensions'] = dimensions
            continue
        
        # IOLIST 시작
        if line.startswith("IOLIST"):
            in_iolist = True
            continue
        
        # ENDIOLIST
        if line.startswith("ENDIOLIST"):
            in_iolist = False
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
                module_type=module_data['type'],
                dimensions=module_data['dimensions'],
                iolist=module_data['iolist'],
                network=module_data['network']
            ))
            module_data = {}
            continue
        
        # IOLIST 데이터 수집
        if in_iolist:
            module_data['iolist'].append(line.strip(';'))
            continue
        
        # NETWORK 데이터 수집
        if in_network:
            module_data['network'].append(line.strip(';'))
            continue

    return modules


# 파일 경로 설정
yal_file = "./example/ami33.yal"  # YAL 파일 경로

# YAL 파일 파싱
modules = parse_yal(yal_file)

# 결과 출력
for module in modules:
    print(module)
