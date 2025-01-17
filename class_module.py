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
