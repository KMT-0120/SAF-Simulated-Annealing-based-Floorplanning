class BTreeNode:
    def __init__(self, module):
        self.module = module  # 현재 노드에 저장된 Module
        self.left = None      # 왼쪽 자식 (오른쪽 인접 모듈)
        self.right = None     # 오른쪽 자식 (위쪽 인접 모듈)

class BStarTree:
    def __init__(self):
        self.root = None  # 트리의 루트 노드

    def insert(self, parent, module, direction):
        """
        B*-Tree에 노드 삽입
        :param parent: 부모 노드 (BTreeNode 객체)
        :param module: 삽입할 Module 객체
        :param direction: 삽입 방향 ('left' 또는 'right')
        """
        new_node = BTreeNode(module)
        if direction == 'left':
            parent.left = new_node
        elif direction == 'right':
            parent.right = new_node
        return new_node

    def traverse_preorder(self, node, result):
        """전위 순회: 현재 노드 → 왼쪽 → 오른쪽"""
        if node:
            result.append(node.module.name)  # 현재 노드 처리
            self.traverse_preorder(node.left, result)
            self.traverse_preorder(node.right, result)

    def traverse_inorder(self, node, result):
        """중위 순회: 왼쪽 → 현재 노드 → 오른쪽"""
        if node:
            self.traverse_inorder(node.left, result)
            result.append(node.module.name)  # 현재 노드 처리
            self.traverse_inorder(node.right, result)

    def traverse_postorder(self, node, result):
        """후위 순회: 왼쪽 → 오른쪽 → 현재 노드"""
        if node:
            self.traverse_postorder(node.left, result)
            self.traverse_postorder(node.right, result)
            result.append(node.module.name)  # 현재 노드 처리
