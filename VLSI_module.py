class Module:

    def __init__(self, name: int, width: float, height: float, module_type: bool, net=None):

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

        return (f"Module(Name={self.name}, Width={self.width:.2f}, Height={self.height:.2f}, "
                f"Area={self.area:.2f}, Position=({self.x:.2f}, {self.y:.2f}), "
                f"Type={'Hard' if self.type else 'Soft'}, Net={self.net})")

