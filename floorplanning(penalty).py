#해당 주석 문단은 지우지말것(특히 GPT 사용시, 지워지는지 또한 확인한 후 다시 추가), 
# GPT요구사항(해당 줄 이외의 주석내용은 GPT 요구사항아니므로 반영하지말것)GPT가 임의로 코드 전체 골격 함수 이름, 변수명 변경하지 말것, 변경하였을경우 자세한 이유와 변경사항 명시할것것

#3/8 iter 5000번마다 plot, SA종료 후 T 변화 추이 그래프 plot 추가
# 사용하지 않는 과거 코드 삭제(adjust_for_overlaps, check_overlap, is_within_bounds)
# plot관련해서 노드 크기/폰트 크기 조정, 노드 간격 조정
# c factor 조정
#3/9 GSRC 적용, 초기해 leftchain -> random으로 변경
#3/15 왼쪽 자식의 배치가 부모노드와 너무 조금 떨어져있던 버그 수정
#3/18 cost에 penalty추가, 정규화는 area_norm와 너무 크게 차이나는것을 확인(가령 HPWL_norm=4, area_norm=2000)해서 area_norm * 500
#     초기, 마지막 비용값을 더 잘 확인하기 위해 get_nomalized_values()함수 → calc_combined_cost에 return_details 인자 추가
#     default_parent를 1000, 1000 대신 전체 module의 넓이*1.2로 설정
#     sample_moves가 버려지는 경우를 대비해 sample_moves 전부를 기억해두고 제일 좋았던 코스트 변화량을 적용
#     left-chain 적용
#3/26 penalty 함수 개선(area_violation, length_violation)
#     cost_scale 100 -> 10, T1_scale_factor 10 -> 1, w 사용빈도가 높아 Global_w로 관리
#     penalty 계산식 윤서ver로 수정, Q-learning 사용
#     x좌표 contour 적용중
#     calc_combined_cost와 get_normalized_values를 통합 → calc_combined_cost(…, return_details=True)


import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import random
import re
import sys
sys.setrecursionlimit(10000)

global Global_w
Global_w=0.66

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
        self.order = None

    def rotate(self):
        """너비와 높이를 교환"""
        self.width, self.height = self.height, self.width

    def set_position(self, x: float, y: float, order: int):
        self.x = x
        self.y = y
        self.order = order

    def __str__(self):
        formatted_net = "\n    ".join(self.net)
        return (f"Module(Name={self.name}, "
                f"Width={self.width:.2f}, Height={self.height:.2f}, "
                f"Area={self.area:.2f}, Position=({self.x:.2f}, {self.y:.2f}), "
                f"Order={self.order}, Type={self.type}, Net=\n    {formatted_net})")

class BTreeNode:
    def __init__(self, module, parent=None):
        self.module = module
        self.left = None
        self.right = None
        self.parent = parent

class ContourNode:
    """Contour segment: [x1, x2) 구간에서 y2 높이"""
    def __init__(self, x1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y2 = y2

class Chip:
    def __init__(self, modules):
        """
        parent가 없는 경우에는 전체 module들의 area합 * 1.2와 동일한
        정사각형을 DefaultParent로 설정합니다.
        """
        self.modules = [m for m in modules if m.type != 'PARENT']

        self.bound = None #paprent 무시할경우
        #self.bount = next((m for m in modules if m.type == 'PARENT'), None) #ami시리즈 아닐경우에는 NONE으로 고정

        if not self.bound:
            total_area = sum(m.area for m in self.modules)
            side = math.sqrt(total_area*1.2) # Default parent의 한 변 길이를 전체 모듈 넓이 합의 제곱근으로 설정
            self.bound = Module(
                name='DefaultParent',
                width=side, # 
                height=side, # 
                module_type='PARENT'
            )

        self.build_b_tree()
        self.contour_line = [] # Initialize contour_line here
        self.max_width = 0
        self.max_height = 0


    def build_b_tree(self):
        if not self.modules:
            self.root = None
            return

        random_mods = self.modules[:]
        random.shuffle(random_mods)

        self.root = BTreeNode(random_mods[0], parent=None)
        for mod in random_mods[1:]:
            node = BTreeNode(mod)
            cands = self.find_possible_parents()
            chosen = random.choice(cands)
            self.attach_node(chosen, node)

    def collect_all_nodes(self):
        if not self.root:
            return []
        result=[]
        queue=[self.root]
        while queue:
            cur=queue.pop(0)
            result.append(cur)
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)
        return result

    def rotate_node(self,node):
        node.module.rotate()

    def move_node(self,node):
        if node.parent is None: # 루트 노드는 이동 불가
            return f"[Op2: Move] Cannot move ROOT node ({node.module.name})."
        self.remove_node_from_parent(node)
        cands = self.find_possible_parents()
        if not cands: # 재삽입할 위치가 없는 경우 (이론적으로는 발생하기 어려움)
             # 만약을 위해 원래대로 복구 시도 (실제로는 더 복잡한 복구 로직 필요)
            # self.attach_node(original_parent, node) # 이 부분은 실제 구현 시 original_parent와 side를 저장해야 함
            return "[Op2: Move] No available spot to re-insert node."
        new_par = random.choice(cands)
        side = self.attach_node(new_par,node)
        return f"[Op2: Move] Moved {node.module.name} under {new_par.module.name} ({side})"

    def remove_node_from_parent(self,node):
        p=node.parent
        if p is None: # 이미 부모가 없는 경우 (루트 노드 등)
            return
        if p.left==node:
            p.left=None
        elif p.right==node:
            p.right=None
        node.parent=None # 노드의 부모 참조도 제거

    def find_possible_parents(self):
        all_nd=self.collect_all_nodes()
        results=[]
        for nd in all_nd:
            if nd.left is None or nd.right is None: # 왼쪽 또는 오른쪽 자식이 없는 노드
                results.append(nd)
        return results

    def attach_node(self,parent,node):
        node.parent=parent # 새 부모 설정
        slots=[]
        if not parent.left:
            slots.append("left")
        if not parent.right:
            slots.append("right")
        if not slots: # 양쪽 자식이 모두 있는 경우 (이론적으로 find_possible_parents에서 걸러짐)
            return "NoSlot" # 오류 또는 예외 상황 처리 필요
        chosen=random.choice(slots)
        if chosen=="left":
            parent.left=node
        else:
            parent.right=node
        return chosen # 어느 쪽에 붙였는지 반환 ("left" or "right")

    def swap_nodes(self,nodeA,nodeB):
        # 단순히 모듈 객체만 교환
        nodeA.module,nodeB.module = nodeB.module,nodeA.module

    def apply_specific_operation(self, op, all_nodes=None):
        """
        Q-learning등에서 rotate/move/swap이 결정된 뒤 실제 수행
        """
        if not all_nodes:
            all_nodes = self.collect_all_nodes()
        if not all_nodes:
            # print("[Warn] No nodes in the tree.") # 사용자 요청에 따라 주석 처리 또는 삭제 가능
            return ("NoNode",None,None) # (메시지, 연산종류, 연산데이터)

        if op=="rotate":
            chosen=random.choice(all_nodes)
            self.rotate_node(chosen)
            msg=f"[Op1: Rotate] Rotated {chosen.module.name}"
            op_data={'op':'rotate','nodeA_name':chosen.module.name}
        elif op=="move":
            chosen=random.choice(all_nodes)
            msg=self.move_node(chosen) # move_node는 이미 메시지를 반환
            op_data={'op':'move','nodeA_name':chosen.module.name}
        elif op=="swap":
            if len(all_nodes)<2:
                msg="[Op3: Swap] Not enough nodes to swap."
                op_data={'op':'swap','nodeA_name':None,'nodeB_name':None} # 스왑 실패 시
            else:
                nodeA,nodeB=random.sample(all_nodes,2)
                self.swap_nodes(nodeA,nodeB)
                msg=f"[Op3: Swap] Swapped {nodeA.module.name} <-> {nodeB.module.name}"
                op_data={'op':'swap','nodeA_name':nodeA.module.name,'nodeB_name':nodeB.module.name}
        else:
            msg="[InvalidOp]"
            op_data={'op':None} # 유효하지 않은 연산
        # print(msg) # <<< 이 부분을 주석 처리 또는 삭제합니다.

        # 연산 후 좌표 재계산
        for nd in all_nodes: # 좌표 초기화
            nd.module.x=0
            nd.module.y=0
            nd.module.order=None
        self.calculate_coordinates()
        return (msg,op,op_data) # (수행결과메시지, 실제수행된연산, 연산데이터)

    def reapply_operation(self, op_data):
        """
        SA에서 이전 상태로 복원하기 위해 연산을 "되돌리는" 것이 아니라,
        저장된 op_data를 기반으로 연산을 "다시 적용"하는 방식.
        실제로는 SA에서 상태 복원은 deepcopy를 사용하므로, 이 함수는
        특정 연산을 재현하거나 디버깅할 때 유용할 수 있음.
        SA의 accept/reject 로직에서는 주로 chip 객체의 deepcopy를 사용.
        여기서는 op_data를 받아 해당 연산을 다시 수행하는 것으로 이해하고 작성.
        """
        if not op_data:
            return ("[NoOpData]", "NoOp")

        all_nd=self.collect_all_nodes()
        def find_node_by_name(nm):
            for nd in all_nd:
                if nd.module.name==nm:
                    return nd
            return None

        op_type=op_data.get('op','NoOp')
        if op_type=='rotate':
            nA=op_data.get('nodeA_name')
            ndA=find_node_by_name(nA)
            if ndA is None:
                return("[Rotate] Node not found",op_type)
            self.rotate_node(ndA) # 회전은 다시 수행해도 동일
            msg=f"[Op1: Rotate] Re-applied rotate on {nA}"
        elif op_type=='move':
            # Move 연산은 되돌리기가 복잡함.
            # 여기서는 단순히 op_data에 명시된 노드를 다시 move 시도하는 것으로 가정.
            # SA에서는 상태 복원을 위해 deepcopy된 이전 chip 객체를 사용해야 함.
            # 이 함수가 SA의 상태 복원용이라면, 이 로직은 부적합.
            # 여기서는 op_data에 있는 노드를 다시 move하는 것으로 해석.
            nA=op_data.get('nodeA_name')
            ndA=find_node_by_name(nA)
            if ndA is None:
                return("[Move] Node not found",op_type)
            msg=self.move_node(ndA) # move_node는 메시지 반환
            msg+=" [Re-applied]"
        elif op_type=='swap':
            nA=op_data.get('nodeA_name')
            nB=op_data.get('nodeB_name')
            ndA=find_node_by_name(nA)
            ndB=find_node_by_name(nB)
            if (not ndA) or (not ndB):
                return("[Swap] Node(s) not found",op_type)
            self.swap_nodes(ndA,ndB) # 스왑은 다시 수행해도 동일
            msg=f"[Op3: Swap] Re-applied swap: {nA} <-> {nB}"
        else:
            msg="[NoOp or invalid op]"
            op_type="NoOp"

        # 연산 후 좌표 재계산
        for nd in all_nd: # 좌표 초기화
            nd.module.x=0
            nd.module.y=0
            nd.module.order=None
        self.calculate_coordinates()
        return(msg,op_type) # (재적용결과메시지, 재적용된연산종류)

    def randomize_b_tree(self,w):
        """B*-Tree 구조를 랜덤하게 재구성하고 비용을 반환"""
        random.shuffle(self.modules) # 모듈 순서 섞기
        self.build_b_tree() # B*-Tree 재구성
        self.calculate_coordinates() # 좌표 계산
        return calc_combined_cost(self.modules,w,chip=self) # 비용 계산

    def calculate_coordinates(self):
        if not self.root:
            return
        self.contour_line=[] # 컨투어 라인 초기화
        self.max_width=0
        self.max_height=0
        all_nd=self.collect_all_nodes() # 모든 노드 좌표 초기화
        for nd in all_nd:
            nd.module.x=0
            nd.module.y=0
            nd.module.order=None
        self._dfs_place_node(self.root,1,0,0) # 루트부터 DFS로 배치 시작
    
    def _dfs_place_node(self,node,order,x_offset,y_offset):
        if not node:
            return order

        # 현재 노드 위치 설정
        node.module.x=x_offset
        node.module.y=y_offset
        node.module.order=order # 배치 순서 기록

        # 현재 노드가 차지하는 x_range와 top_y
        x1=x_offset
        x2=x_offset+node.module.width
        top_y=y_offset+node.module.height

        # 컨투어 라인 업데이트 및 최대 너비/높이 갱신
        self.insert_contour_segment(x1,x2,top_y)
        if x2>self.max_width:
            self.max_width=x2
        if top_y>self.max_height:
            self.max_height=top_y
        order+=1

        # 왼쪽 자식 배치 (부모의 오른쪽에 배치)
        if node.left:
            lx=node.module.x+node.module.width # 왼쪽 자식의 x 시작점은 부모의 x + 부모 너비
            rx=lx+node.left.module.width # 왼쪽 자식의 x 끝점
            ly=self.update_contour(lx,rx) # 해당 x구간에서 가능한 y좌표
            order=self._dfs_place_node(node.left,order,lx,ly)
        
        # 오른쪽 자식 배치 (부모의 아래쪽에 배치)
        if node.right:
            rx_s=node.module.x # 오른쪽 자식의 x 시작점은 부모의 x와 동일
            rx_e=rx_s+node.right.module.width # 오른쪽 자식의 x 끝점
            ry_s=self.update_contour(rx_s,rx_e) # 해당 x구간에서 가능한 y좌표
            order=self._dfs_place_node(node.right,order,rx_s,ry_s)

        return order 

    def update_contour(self,x1,x2):
        """주어진 x_range [x1, x2)에 대해 배치될 수 있는 가장 낮은 y값 찾기"""
        base_y=0
        for seg in self.contour_line:
            # 겹치는 구간이 있는 경우
            if not(seg.x2<=x1 or seg.x1>=x2):
                base_y=max(base_y,seg.y2)
        return base_y

    def insert_contour_segment(self,x1,x2,new_top):
        """새로운 모듈 배치 후 컨투어 라인 업데이트"""
        updated=[]
        i=0
        while i<len(self.contour_line):
            seg=self.contour_line[i]
            # 새 세그먼트와 겹치지 않는 경우
            if seg.x2<=x1 or seg.x1>=x2:
                updated.append(seg)
            else: # 겹치는 경우
                # 기존 세그먼트가 새 세그먼트를 포함하는 경우 (양쪽으로 쪼개짐)
                if seg.x1<x1 and seg.x2>x2:
                    updated.append(ContourNode(seg.x1,x1,seg.y2))
                    updated.append(ContourNode(x2,seg.x2,seg.y2))
                # 기존 세그먼트의 왼쪽 부분만 남는 경우
                elif seg.x1<x1 and seg.x2<=x2:
                    updated.append(ContourNode(seg.x1,x1,seg.y2))
                # 기존 세그먼트의 오른쪽 부분만 남는 경우
                elif seg.x1>=x1 and seg.x2>x2:
                    updated.append(ContourNode(x2,seg.x2,seg.y2))
                # 그 외 (새 세그먼트가 기존 세그먼트를 완전히 덮는 등)는 기존 세그먼트 삭제
            i+=1
        updated.append(ContourNode(x1,x2,new_top)) # 새 세그먼트 추가
        updated.sort(key=lambda s:s.x1) # x1 기준으로 정렬
        self.contour_line=updated


    def plot_b_tree(self, iteration=None):
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(24,12)) # Plot 크기 조정

        # 물리적 배치 플롯 (ax1)
        self._plot_node(self.root,ax1) # 모듈 그리기
        # 경계 상자 그리기 (DefaultParent 또는 PARENT 모듈 기준)
        bound_rect=plt.Rectangle((0,0),self.bound.width,self.bound.height,
                                 edgecolor='red',facecolor='none',lw=2, label='Chip Boundary')
        ax1.add_patch(bound_rect)
        ax1.legend()


        if iteration is not None:
            ax1.set_title(f"B*-Tree Physical Placement (Iter={iteration})")
        else:
            ax1.set_title("B*-Tree Physical Placement")
        ax1.set_xlabel("X-coordinate")
        ax1.set_ylabel("Y-coordinate")
        # 축 범위 설정 시 약간의 여유를 둠
        ax1.set_xlim(-50,max(self.max_width, self.bound.width)+200)
        ax1.set_ylim(-50,max(self.max_height, self.bound.height)+200)
        ax1.set_aspect('equal', adjustable='box') # 가로세로 비율 유지

        # B*-Tree 구조 플롯 (ax2)
        G=nx.DiGraph()
        pos={} # 노드 위치 저장
        # 트리 구조를 그리기 위한 초기 x 위치 (루트 노드는 중앙에 가깝게)
        # 너비 계산을 위해 임시로 호출할 수 있지만, 여기서는 대략적인 값으로 시작
        self.add_edges(self.root,None,0,0,G,pos) # 루트부터 시작

        nx.draw(G, pos, ax=ax2, with_labels=True, node_color="lightblue",
                edge_color="black", node_size=1500, font_size=8,  # 노드/폰트 크기 조정
                arrows=True, arrowstyle='->', arrowsize=10)
        if iteration is not None:
            ax2.set_title(f"B*-Tree Structure (Iter={iteration})")
        else:
            ax2.set_title("B*-Tree Structure")
        plt.tight_layout() # 서브플롯 간 간격 자동 조정


    def _plot_node(self,node,ax):
        if not node:
            return
        m=node.module
        rect=plt.Rectangle((m.x,m.y),m.width,m.height,
                           edgecolor='blue',facecolor='lightblue',fill=True,lw=1) # 선 두께 조정
        ax.add_patch(rect)
        ax.text(m.x+m.width/2,m.y+m.height/2,
                f'{m.name}\n#{m.order}',ha='center',va='center',fontsize=7) # 폰트 크기 조정
        self._plot_node(node.left,ax)
        self._plot_node(node.right,ax)

    def add_edges(self,node,parent_name,depth,x_pos,G,pos):
        """
        트리 구조를 시각화하기 위해 NetworkX 그래프에 노드와 엣지를 추가하고,
        계층적으로 노드 위치(pos)를 계산합니다.
        x_pos는 현재 서브트리가 그려질 x축 시작 위치를 나타냅니다.
        반환값은 현재 서브트리 이후에 다음 서브트리가 그려질 x축 시작 위치입니다.
        """
        if not node:
            return x_pos # 현재 x 위치를 그대로 반환

        node_name=node.module.name
        # pos 계산: x_pos는 왼쪽부터의 상대적 위치, depth는 y축 위치
        # 실제 그릴 때는 x_pos에 일정한 간격을 곱하여 노드들이 겹치지 않도록 함
        # y 위치는 깊이에 따라 음수 값으로 설정하여 위에서 아래로 그려지도록 함
        node_x_coordinate = x_pos # 왼쪽 자식 먼저 그리고, 그 다음 현재 노드
        
        # 왼쪽 자식 먼저 재귀 호출하여 왼쪽 서브트리의 너비를 확보
        # 왼쪽 자식이 차지하는 x 공간을 계산하고, 그 이후에 현재 노드를 배치
        current_x_start = x_pos
        if node.left:
            current_x_start = self.add_edges(node.left, node_name, depth + 1, x_pos, G, pos)
        
        # 현재 노드 위치 설정 (왼쪽 자식 서브트리 이후)
        # pos[node_name] = (current_x_start * 20, -depth * 30) # 간격 조정
        pos[node_name] = (current_x_start, -depth * 2) # 간격 조정 (이전 버전과 유사하게)
        G.add_node(node_name,label=node.module.name)
        if parent_name:
            G.add_edge(parent_name,node_name)

        # 현재 노드 이후의 x 위치 (오른쪽 자식을 위해)
        next_x_pos = current_x_start + 1 # 현재 노드가 1만큼의 x 공간을 차지한다고 가정하고 다음 위치 설정

        if node.right:
            # 오른쪽 자식은 현재 노드 바로 다음 또는 약간 떨어진 위치에서 시작
            next_x_pos = self.add_edges(node.right, node_name, depth + 1, next_x_pos, G, pos)
            return next_x_pos # 오른쪽 자식까지 그린 후의 x 위치 반환
        
        return next_x_pos # 오른쪽 자식이 없으면 현재 노드 다음 위치 반환


# ─────────────────────────────────────────────────────────────
# HPWL, 면적, penalty 함수들
# ─────────────────────────────────────────────────────────────

def calculate_hpwl(modules):
    net_dict={} # 각 net에 연결된 모듈들을 저장
    for m in modules:
        for net_name in m.net:
            if net_name not in net_dict:
                net_dict[net_name]=[]
            net_dict[net_name].append(m) # 모듈 객체 자체를 추가
    if not net_dict: # 연결된 net이 없는 경우
        return 0
    total_hpwl=0
    for net,mods in net_dict.items():
        if not mods: continue # 해당 net에 연결된 모듈이 없으면 스킵
        # 각 net에 연결된 모듈들의 중심점 대신, 모듈의 경계 사용
        min_x=min(mm.x for mm in mods)
        max_x=max(mm.x+mm.width for mm in mods)
        min_y=min(mm.y for mm in mods)
        max_y=max(mm.y+mm.height for mm in mods)
        net_hpwl=(max_x - min_x)+(max_y - min_y)
        total_hpwl+=net_hpwl
    return total_hpwl

def calculate_total_area(modules):
    if not modules:
        return (0,0,0) # (width, height, area)
    # 모든 모듈을 포함하는 최소 경계 상자(bounding box) 계산
    min_x=min(m.x for m in modules)
    max_x=max(m.x+m.width for m in modules)
    min_y=min(m.y for m in modules)
    max_y=max(m.y+m.height for m in modules)
    width=(max_x - min_x)
    height=(max_y - min_y)
    return (width,height,(width*height))

def calc_combined_cost(modules, w=Global_w, chip=None, r=1.0, return_all=False):
    """
    return_all=False => float( cost )
    return_all=True  => ( cost, area_norm, hpwl_norm, penalty_norm ) 튜플
    """
    cost_scale=10 # 비용 스케일 조정 (이전 100 -> 10)
    if not modules:
        if return_all:
            return (0.0, 0.0, 0.0, 0.0)
        else:
            return 0.0

    # 정규화를 위한 기준값 계산
    base_scale=sum(m.area for m in modules) # 모든 모듈의 실제 면적 합
    if base_scale<1e-12: # 0으로 나누는 것 방지
        base_scale=1.0
    # HPWL 정규화를 위한 기준값 (net이 있는 모듈들의 면적 합의 제곱근)
    net_area_sum=sum(m.area for m in modules if m.net)
    if net_area_sum<1e-12:
        net_area_sum=base_scale # net이 없거나 매우 작으면 전체 면적 기준으로

    w_,h_,area_bb=calculate_total_area(modules) # 현재 배치의 Bounding Box 면적
    hpwl=calculate_hpwl(modules)

    # penalty 계산 (chip 경계를 벗어나는 경우)
    if chip and chip.bound: # chip 객체와 경계 정보가 있을 때만 penalty 계산
        bound_w = chip.bound.width
        bound_h = chip.bound.height
        # 현재 배치의 전체 너비와 높이 (calculate_total_area에서 이미 계산됨)
        # w_bb, h_bb, _ = calculate_total_area(modules) # 중복 계산 피하기 위해 위에서 받은 값 사용
        w_bb, h_bb = w_, h_

        # Area violation (윤서 ver penalty)
        # 전체 bounding box가 chip 경계를 얼마나 벗어났는지
        area_violation = 0.0
        if w_bb > bound_w and h_bb > bound_h: # 둘 다 초과
            area_violation = (w_bb * h_bb) - (bound_w * bound_h)
        elif w_bb > bound_w: # 너비만 초과
            area_violation = (w_bb - bound_w) * h_bb # 초과된 너비 * 현재 높이
        elif h_bb > bound_h: # 높이만 초과
            area_violation = (h_bb - bound_h) * w_bb # 초과된 높이 * 현재 너비


        # Length violation (윤서 ver penalty)
        # 개별 모듈이 chip 경계를 얼마나 벗어났는지 (제곱으로 패널티 강화)
        length_violation=0.0
        for mm in modules:
            # 오른쪽 경계 초과
            if mm.x + mm.width > bound_w:
                length_violation += ( (mm.x + mm.width) - bound_w )**2
            # 위쪽 경계 초과
            if mm.y + mm.height > bound_h:
                length_violation += ( (mm.y + mm.height) - bound_h )**2
            # 왼쪽 경계 초과 (x < 0)
            if mm.x < 0:
                length_violation += (mm.x)**2 # (-x)^2 = x^2
            # 아래쪽 경계 초과 (y < 0)
            if mm.y < 0:
                length_violation += (mm.y)**2


        penalty_sum=area_violation+length_violation
    else: # chip 경계 정보가 없으면 penalty 없음
        penalty_sum=0.0

    # 정규화된 값 계산
    area_norm   = area_bb/base_scale
    hpwl_norm   = hpwl/(2*math.sqrt(net_area_sum)) if net_area_sum > 1e-9 else hpwl # 분모 0 방지
    penalty_norm= penalty_sum/base_scale # penalty도 base_scale로 정규화

    # 정규화된 값에 가중치 적용 (이전 버전의 스케일링 유지)
    area_norm   *=500
    penalty_norm*=50 # penalty 가중치

    # 최종 비용 계산 (w: area 가중치, 1-w: hpwl 가중치, r: penalty 가중치)
    cost_val= w*area_norm + (1-w)*hpwl_norm + r*penalty_norm
    cost_val*= cost_scale # 전체 비용 스케일링

    if return_all:
        return (cost_val, area_norm, hpwl_norm, penalty_norm)
    else:
        return cost_val

# ─────────────────────────────────────────────────────────────
# Q-Learning / fast_sa
# ─────────────────────────────────────────────────────────────

Q_TABLE={} # Q-테이블 (상태, 액션별 Q값 저장)
ACTIONS=["rotate","move","swap"] # 가능한 액션
epsilon=0.3 # Epsilon-greedy 탐색 확률 (무작위 액션 선택)
alpha=0.5   # 학습률 (Q값 업데이트 시 이전 값 반영 비율)
gamma=0.9   # 할인율 (미래 보상 가치 반영 비율)

def get_state_key(chip):
    """현재 chip 상태를 나타내는 key 생성 (비용 기반으로 단순화)"""
    # 비용을 소수점 한 자리까지 반올림하여 상태 구분 (더 정교한 상태 정의 필요 가능성)
    c_val=calc_combined_cost(chip.modules, w=Global_w, chip=chip, r=1.0, return_all=False)
    return round(c_val,1) # 비용을 상태 키로 사용

def select_action_by_q(state_key):
    """Q-테이블과 epsilon-greedy 전략에 따라 액션 선택"""
    if random.random()<epsilon: # epsilon 확률로 무작위 액션 선택 (탐험)
        return random.randrange(len(ACTIONS))
    else: # 그 외에는 Q값이 가장 높은 액션 선택 (활용)
        if state_key not in Q_TABLE: # 처음 보는 상태면 Q값 초기화
            Q_TABLE[state_key]=[0.0,0.0,0.0]
        arr=Q_TABLE[state_key]
        return arr.index(max(arr)) # Q값이 가장 큰 액션의 인덱스 반환

def update_q_value(s_key,a_idx,reward,s_next_key):
    """Q-러닝 업데이트 규칙에 따라 Q값 갱신"""
    if s_key not in Q_TABLE:
        Q_TABLE[s_key]=[0.0,0.0,0.0]
    if s_next_key not in Q_TABLE: # 다음 상태도 Q-테이블에 없으면 초기화
        Q_TABLE[s_next_key]=[0.0,0.0,0.0]
    
    q_old=Q_TABLE[s_key][a_idx] # 현재 상태-액션의 Q값
    max_next=max(Q_TABLE[s_next_key]) # 다음 상태에서 가능한 최대 Q값
    
    # Q-러닝 업데이트 공식
    q_new=q_old+alpha*(reward+gamma*max_next-q_old)
    Q_TABLE[s_key][a_idx]=q_new


def fast_sa(chip,max_iter=5000,P=0.99,c=20,w=Global_w,sample_moves=10,r=1.0):
    import copy,math,random

    # (A) T1 (초기 온도) 추정
    orig_st=copy.deepcopy(chip) # 초기 상태 저장
    orig_cost=calc_combined_cost(chip.modules,w,chip=chip,r=r,return_all=False)
    up_diffs=[] # 비용이 증가한 경우의 dE 저장 리스트
    
    # 초기 온도를 결정하기 위해 여러 번의 랜덤 이동 시도
    for _ in range(sample_moves+10): # sample_moves보다 많이 시도
        s_key=get_state_key(chip) # 현재 상태 키
        a_idx=select_action_by_q(s_key) # Q러닝 기반 액션 선택
        action_name=ACTIONS[a_idx]

        old_c=calc_combined_cost(chip.modules,w,chip=chip,r=r,return_all=False)
        msg_tmp,op_tmp,op_data_tmp=chip.apply_specific_operation(action_name) # 액션 적용
        new_c=calc_combined_cost(chip.modules,w,chip=chip,r=r,return_all=False)
        dE=new_c-old_c # 비용 변화량

        rew=(old_c-new_c) # 보상 (비용 감소가 큰 쪽이 높은 보상)
        s_next=get_state_key(chip) # 다음 상태 키
        update_q_value(s_key,a_idx,rew,s_next) # Q값 업데이트

        if dE>0: # 비용이 증가한 경우만
            up_diffs.append(dE)
        chip=copy.deepcopy(orig_st) # 상태 원복 (T1 추정을 위함이므로 실제 이동은 아님)

    if up_diffs:
        delta_avg=sum(up_diffs)/len(up_diffs) # 비용 증가량 평균
    else: # 비용 증가가 한 번도 없었으면 기본값 사용
        delta_avg=1.0 
    if delta_avg<1e-12: # 매우 작은 경우 대비
        delta_avg=1.0

    T1_scale_factor=1.0 # T1 스케일 조정 (이전 10 -> 1)
    T1=abs(delta_avg/math.log(P))*T1_scale_factor # 초기 온도 T1 계산
    print(f"Initial T1={T1:.3f}")

    best_chip=copy.deepcopy(chip) # 전체 SA 과정에서 최적 해 저장
    best_cost=orig_cost
    cur_cost =orig_cost # 현재 해의 비용

    temps=[] # 온도 변화 기록
    for n in range(1,max_iter+1): # 최대 반복 횟수만큼 SA 진행
        st_cp=copy.deepcopy(chip) # 현재 상태 복사 (여러 move 시도용)
        oldc=cur_cost # 현재 반복 시작 시점의 비용 (루프 내에서 기준)

        # sample_moves 만큼 다양한 연산을 시도하고 가장 좋았던 변화를 선택
        cost_diffs=[] # 각 시도별 비용 변화량 절대값 저장
        best_local_cost=float('inf') # sample_moves 중 가장 좋았던 비용
        best_op_data=None # 가장 좋았던 연산 정보

        for _ in range(sample_moves):
            s_key2=get_state_key(st_cp) # 현재 st_cp의 상태
            a_idx2=select_action_by_q(s_key2) # Q러닝 액션 선택
            act2=ACTIONS[a_idx2]

            old_c2=calc_combined_cost(st_cp.modules,w,chip=st_cp,r=r,return_all=False)
            msg2,op2,opd2=st_cp.apply_specific_operation(act2) # 연산 적용
            new_c2=calc_combined_cost(st_cp.modules,w,chip=st_cp,r=r,return_all=False)

            rew2=(old_c2-new_c2) # 보상
            s_next2=get_state_key(st_cp) # 다음 상태
            update_q_value(s_key2,a_idx2,rew2,s_next2) # Q값 업데이트

            dE_from_iter_start=new_c2-oldc # 이번 SA 반복 시작 시점(oldc)과의 비용차
            cost_diffs.append(abs(dE_from_iter_start)) # 비용 변화량 절대값

            if new_c2<best_local_cost: # 더 좋은 해를 찾으면
                best_local_cost=new_c2
                best_op_data=opd2 # 해당 연산 정보 저장
            
            st_cp=copy.deepcopy(chip) # 다음 sample_move를 위해 chip 상태로 원복

        # 온도 업데이트 (냉각 스케줄)
        if cost_diffs: # 비용 변화가 있었으면
            delta_cost=max(sum(cost_diffs)/len(cost_diffs),1e-6) # 평균 비용 변화량 (최소값 보장)
        else: # 변화가 없었으면 작은 값 사용
            delta_cost=1e-6

        if n==1: # 첫 반복은 T1 사용
            T=T1
        elif 2<=n<=300: # 초기에는 빠르게 냉각 (c factor 사용)
            T=max((T1*delta_cost)/(n*c),1e-6) # 0 방지
        else: # 이후에는 서서히 냉각
            T=max((T1*delta_cost)/n,1e-6) # 0 방지
        temps.append(T) # 온도 기록

        # 가장 좋았던 연산(best_op_data)을 실제 chip에 적용
        old_chip=copy.deepcopy(chip) # 이전 chip 상태 저장 (reject 대비)
        old_cost=cur_cost # 이전 비용 저장
        
        # best_op_data가 None일 수 있음 (sample_moves에서 유효한 연산이 없었을 경우)
        if best_op_data:
            re_msg,re_op=chip.reapply_operation(best_op_data) # 저장된 최적 연산 적용
        else: # sample_moves에서 유효한 연산이 없었다면, 랜덤 연산 한 번 더 시도
            # print("[Warning] No best_op_data found from sample_moves. Applying a random op.")
            s_rand_key = get_state_key(chip)
            a_rand_idx = select_action_by_q(s_rand_key)
            act_rand = ACTIONS[a_rand_idx]
            re_msg, re_op, best_op_data = chip.apply_specific_operation(act_rand)


        new_c=calc_combined_cost(chip.modules,w,chip=chip,r=r,return_all=False)
        dE=new_c-old_cost # 비용 변화량 (이동 후 - 이동 전)

        # Metropolis 기준에 따라 해 수용 여부 결정
        if dE<0: # 비용 감소: 항상 수용
            cur_cost=new_c
            if new_c<best_cost: # 전체 최적해보다 좋으면 업데이트
                best_cost=new_c
                best_chip=copy.deepcopy(chip)
            acc_str="ACCEPT (better)"
            acc_prob=1.0
        else: # 비용 증가: 확률적으로 수용
            if T<1e-12: # 온도가 매우 낮으면 거의 수용 안함
                acc_prob=0.0
            else:
                acc_prob=math.exp(-abs(dE)/T) # 수용 확률
            
            if random.random()<acc_prob: # 확률에 따라 수용
                cur_cost=new_c
                # 비용이 증가했지만 수용된 경우도 best_cost와 비교할 필요는 없음
                # best_cost는 항상 가장 낮은 비용을 유지해야 함
                # 하지만, SA 특성상 일시적으로 나쁜 해를 탐색할 수 있으므로,
                # 이 경우 best_chip을 업데이트 하지 않음.
                # (주석: 만약 비용이 증가한 해가 전체 최적해보다 좋을 수는 없음. dE>0 이므로)
                acc_str="ACCEPT (worse)"
            else: # 수용 안함: 이전 상태로 복귀
                chip=copy.deepcopy(old_chip)
                cur_cost=old_cost # 비용도 이전 비용으로
                acc_str="REJECT"

        # 로그 출력 (너무 자주 출력하면 느려지므로 조절 필요)
        if n % 100 == 0 or n == 1 or n == max_iter : # 처음, 마지막, 100번째마다 출력
            print(f"[Iter={n:4d}] T={T:8.4f} | Cost={cur_cost:8.3f} (Best={best_cost:8.3f}) | dE={dE:8.3f} | Prob={acc_prob:6.4f} | {acc_str} | Op: {best_op_data.get('op') if best_op_data else 'N/A'}")
            # print(f"ReAppMsg={re_msg}") # re_msg는 디버깅 시 유용

        # 특정 반복마다 현재 best_chip 플롯 (디버깅용)
        # if n%2000==0:
        #     best_chip.plot_b_tree(iteration=n)
        #     plt.show()

    # SA 종료 후 온도 변화 그래프 플롯
    plt.figure(figsize=(10,6)) 
    plt.plot(range(1,max_iter+1),temps)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.title("Temperature vs. Iteration in SA")
    plt.grid(True)
    plt.show()

    return best_chip # 최종적으로 가장 좋았던 해 반환

def partial_sa_for_initial(chip, pre_iter=500):
    """
    부분적으로 짧은 SA를 먼저 돌려서,
    완전 무작위 초기 해보다는 어느 정도 정돈된 상태를 얻기 위한 함수.
    """
    print("\n[Info] --- Running partial SA for initial layout ---\n")
    improved_chip = fast_sa(
        chip,
        max_iter=pre_iter, # 짧은 반복
        P=0.95, # 초기 수용 확률
        c=50,  # 냉각 상수 (조금 더 빠르게 냉각되도록 c값을 줄임, 원본 SA의 c와 다름)
        w=Global_w,
        sample_moves=15, # 초기 해 개선에는 sample_moves를 줄여도 될 수 있음
        r=1.0 # penalty 가중치
    )
    print("[Info] --- Partial SA ended. Using improved initial layout ---\n")
    return improved_chip

# ─────────────────────────────────────────────────────────────
# 4. 파일 파싱(Yal, GSRC)
# ─────────────────────────────────────────────────────────────

def parse_yal(file_path):
    modules=[]
    with open(file_path,'r') as f:
        lines=f.readlines()
    module_data={}
    in_network=False # NETWORK 섹션 내부인지 여부
    for line in lines:
        line=line.strip()
        if not line or line.startswith("/*") or line.startswith("*"): # 빈 줄 또는 주석 건너뛰기
            continue
        
        # MODULE 시작
        if line.startswith("MODULE"):
            parts = line.split()
            if len(parts) > 1:
                module_name=parts[1].strip(';')
                module_data={'name':module_name,'net':[]} # 새 모듈 데이터 초기화
            continue
        
        # TYPE 정보
        if line.startswith("TYPE"):
            parts = line.split()
            if len(parts) > 1:
                module_type=parts[1].strip(';')
                module_data['type']=module_type
            continue

        # DIMENSIONS 정보 (사각형 가정)
        if line.startswith("DIMENSIONS"):
            # "DIMENSIONS (0,0) (0,H) (W,0) (W,H);" 형태 또는 "DIMENSIONS W H;" 형태 고려
            # 여기서는 W, H만 주어지는 간단한 형태나, 꼭지점 좌표가 주어지는 형태 모두 처리 시도
            dim_str = line.replace("DIMENSIONS","").replace(";","").strip()
            coords = []
            # (x,y) 쌍으로 된 좌표 파싱 시도
            if '(' in dim_str and ')' in dim_str:
                points_str = re.findall(r'\([\d\.\s,]+\)', dim_str)
                for p_str in points_str:
                    try:
                        x, y = map(float, p_str.strip('()').split(','))
                        coords.append(x)
                        coords.append(y)
                    except ValueError:
                        # x,y 쌍이 아닌 경우 (예: "W H" 형태) 대비
                        pass
            
            if not coords: # (x,y) 쌍 파싱 실패 시, 공백으로 구분된 숫자들로 파싱 시도
                dims=list(map(float,dim_str.split()))
            else:
                dims = coords

            if len(dims) == 2: # 너비, 높이만 주어진 경우
                module_data['width']=dims[0]
                module_data['height']=dims[1]
            elif len(dims) >= 4: # 좌표가 주어진 경우 (최소 2점 이상)
                x_coords=[dims[i] for i in range(0,len(dims),2)]
                y_coords=[dims[i+1] for i in range(0,len(dims),2)]
                if x_coords and y_coords:
                    module_data['width']=max(x_coords)-min(x_coords)
                    module_data['height']=max(y_coords)-min(y_coords)
                else: # 파싱 실패 시 기본값
                    module_data['width']=0
                    module_data['height']=0
            else: # 그 외 잘못된 형식
                module_data['width']=0
                module_data['height']=0
            continue
            
        # NETWORK 섹션 시작/종료
        if line.startswith("NETWORK"):
            in_network=True
            continue
        if line.startswith("ENDNETWORK"):
            in_network=False
            continue
            
        # MODULE 종료 -> 모듈 객체 생성 및 리스트 추가
        if line.startswith("ENDMODULE"):
            if 'name' in module_data and 'width' in module_data and 'height' in module_data and 'type' in module_data:
                 modules.append(Module(
                    name=module_data['name'],
                    width=module_data['width'],
                    height=module_data['height'],
                    module_type=module_data['type'],
                    net=module_data.get('net',[]) # net 정보가 없을 수도 있음
                ))
            module_data={} # 다음 모듈을 위해 초기화
            continue
            
        # NETWORK 섹션 내부의 net 정보
        if in_network:
            # "NET_NAME;" 형태이므로 ';' 제거
            module_data['net'].append(line.strip(';'))
            continue
            
    return modules

def parse_gsrc_blocks(blocks_file):
    modules=[]
    # 정규 표현식: "sb_숫자 hardrectilinear 개수 (x1,y1) (x2,y2) ..."
    # 그룹1: 블록이름(sb_숫자), 그룹2: 꼭지점개수, 그룹3: 좌표들 문자열
    pattern=re.compile(r"^(sb\d+)\s+\S+\s+(\d+)\s+(.*)$") # \S+는 hardrectilinear 등 타입부분
    with open(blocks_file,'r') as f:
        lines=f.readlines()
    for line in lines:
        line=line.strip()
        # 주석, 빈 줄, 헤더 정보 등 건너뛰기
        if not line or line.startswith('#') or line.startswith('UCSC'):
            continue
        # 터미널(패드) 정보는 현재 처리 안 함 (필요시 추가)
        if line.startswith('p') and 'terminal' in line:
            continue

        match=pattern.match(line)
        if match:
            blk_name=match.group(1)
            # num_pts=int(match.group(2)) # 꼭지점 개수는 직접 사용 안함
            coords_str=match.group(3)
            
            # 좌표 문자열에서 (x, y) 쌍 추출
            # 예: "(0,0) (0,1000) (1000,0) (1000,1000)"
            cpat=re.compile(r"\(([\-\d\.]+)\s*,\s*([\-\d\.]+)\)") # 소수점, 음수 지원
            found_coords=cpat.findall(coords_str)
            
            xs,ys=[],[]
            for (sx,sy) in found_coords:
                try:
                    xs.append(float(sx))
                    ys.append(float(sy))
                except ValueError:
                    print(f"[Warning] Could not parse coordinates for {blk_name}: ({sx},{sy})")
                    continue # 파싱 오류 시 해당 좌표 스킵
            
            if not xs or not ys: # 유효한 좌표가 없으면 스킵
                print(f"[Warning] No valid coordinates found for {blk_name}")
                continue

            minx,maxx=min(xs),max(xs)
            miny,maxy=min(ys),max(ys)
            w_=maxx-minx
            h_=maxy-miny
            modules.append(Module(blk_name,w_,h_,'BLOCK',net=[])) # 타입은 'BLOCK'으로 고정
    return modules

def parse_gsrc_nets(nets_file,modules):
    name_map={m.name:m for m in modules} # 모듈 이름으로 모듈 객체 빠르게 찾기
    i=0
    with open(nets_file,'r') as f:
        lines=f.readlines()
    
    net_id_counter=0 # 고유한 net 이름을 생성하기 위한 카운터
    while i<len(lines):
        line=lines[i].strip()
        # 주석, 빈 줄, 헤더 정보 등 건너뛰기
        if not line or line.startswith('#') or line.startswith('UCLA'):
            i+=1
            continue
        
        # NetDegree 정보 파싱
        if line.startswith('NetDegree'):
            parts=line.split(':')
            deg_str=(parts[1].strip() if len(parts)>1 else "0")
            try:
                deg=int(deg_str)
            except ValueError: # 숫자 변환 실패 시
                print(f"[Warning] Could not parse NetDegree: {line}")
                deg=0 # 0으로 처리하고 다음으로
            
            net_name=f"Net{net_id_counter}" # 고유 net 이름 부여
            net_id_counter+=1
            
            pins_in_current_net=[] # 현재 net에 연결된 핀(모듈) 이름들
            for _ in range(deg): # degree만큼 다음 줄들을 읽어 핀 정보 파싱
                i+=1
                if i>=len(lines): break # 파일 끝 도달 방지
                pin_ln=lines[i].strip()
                pin_prt=pin_ln.split() # 예: "sb123 B ..." -> 첫번째가 핀/모듈 이름
                if pin_prt:
                    pin_name=pin_prt[0]
                    if pin_name in name_map: # 해당 모듈이 존재하는 경우
                        pins_in_current_net.append(pin_name)
                    # else: 터미널(패드)인 경우 등은 현재 모듈 리스트에 없을 수 있음 (무시)
            
            # 수집된 핀(모듈)들에 net 정보 추가
            for pn_name in pins_in_current_net:
                name_map[pn_name].net.append(net_name)
        i+=1
    return modules

# ─────────────────────────────────────────────────────────────
# 5. 메인 실행
# ─────────────────────────────────────────────────────────────
if __name__=="__main__":
    # Yal 파일 또는 GSRC 파일 선택 사용
    # --- Yal 예제 ---
    #blocks_file="./example/ami49.yal"
    #modules=parse_yal(blocks_file)

    #--- GSRC 예제 ---
    blocks_file="./example/n300.blocks"
    nets_file  ="./example/n300.nets"
    modules=parse_gsrc_blocks(blocks_file)
    modules=parse_gsrc_nets(nets_file,modules) # nets 정보는 blocks 파싱 후에 추가

    if not modules:
        print("[Error] No modules loaded. Exiting.")
        sys.exit(1)


    # [A] 초기 Chip 생성 -> B*-Tree 랜덤 구성 및 초기 좌표 계산
    chip=Chip(modules)
    chip.calculate_coordinates() # 초기 좌표 계산
    
    # 초기 상태 플롯
    print("Plotting initial random B*-Tree layout...")
    chip.plot_b_tree(iteration="Initial_Random")
    plt.show()


    # 초기 비용 계산 및 출력
    init_cost, aN,hN,pN = calc_combined_cost(chip.modules, w=Global_w, chip=chip, r=1.0, return_all=True)
    w_init,h_init,area_init=calculate_total_area(chip.modules)
    hpwl_init =calculate_hpwl(chip.modules)
    print("=== Initial Random Chip State ===")
    print(f"BoundingBox: W={w_init:.2f}, H={h_init:.2f}, Area={area_init:.2f}")
    print(f"HPWL (absolute)             = {hpwl_init:.2f}")
    print(f"Normalized Area             = {aN:.3f}")
    print(f"Normalized HPWL             = {hN:.3f}")
    print(f"Normalized Penalty          = {pN:.3f}")
    print(f"Initial Cost (w={Global_w:.2f}, r=1.0) = {init_cost:.3f}")


    # [B] 부분적 SA(짧은 SA)를 통한 초기 배치 개선 (선택적)
    use_partial_sa = input("Run partial SA for initial layout improvement? (y/n): ")
    if use_partial_sa.lower().startswith('y'):
        chip = partial_sa_for_initial(chip, pre_iter=500) # 반복 횟수 조정 가능 (예: 500~2000)
        print("Plotting layout after partial SA...")
        chip.plot_b_tree(iteration="After_Partial_SA")
        plt.show()

        # 부분 SA 후 비용 재계산 및 출력
        partial_sa_cost, paN,phN,ppN = calc_combined_cost(chip.modules, w=Global_w, chip=chip, r=1.0, return_all=True)
        pw,ph,parea=calculate_total_area(chip.modules)
        phpwl=calculate_hpwl(chip.modules)
        print("=== Chip State After Partial SA ===")
        print(f"BoundingBox: W={pw:.2f}, H={ph:.2f}, Area={parea:.2f}")
        print(f"HPWL (absolute)             = {phpwl:.2f}")
        print(f"Normalized Area             = {paN:.3f}")
        print(f"Normalized HPWL             = {phN:.3f}")
        print(f"Normalized Penalty          = {ppN:.3f}")
        print(f"Cost after Partial SA       = {partial_sa_cost:.3f}")


    # [C] 메인 SA 실행 여부
    ans=input("Proceed with Full FastSA (Q-Learning) optimization? (y/n): ")
    if ans.lower().startswith('y'):
        best_chip=fast_sa(chip, # 이전 단계의 chip 객체를 사용
                          max_iter=8000,    # 반복 횟수 (예: 5000 ~ 20000)
                          P=0.95,            # 초기 수용 확률
                          c=100,             # 냉각 스케줄 상수 (조정 필요)
                          w=Global_w,        # Area vs HPWL 가중치
                          sample_moves=20,   # 각 온도 스텝에서 시도할 이동 수
                          r=20.0)             # Penalty 가중치

        # 최종 결과 비용 계산 및 출력
        final_cost, faN, fhN, fpN = calc_combined_cost(best_chip.modules, w=Global_w, chip=best_chip, r=1.0, return_all=True)
        fw,fh,farea=calculate_total_area(best_chip.modules)
        fhpwl=calculate_hpwl(best_chip.modules)

        total_mod_area=sum(m.area for m in best_chip.modules)
        if farea>1e-9: # 0으로 나누기 방지
            ds=((farea-total_mod_area)/farea)*100 # Dead Space 비율 (%)
        else:
            ds=0.0

        print("\n=== Full FastSA (Q-Learning) Finished ===")
        print(f"Final BoundingBox: W={fw:.2f}, H={fh:.2f}, Area={farea:.2f}")
        print(f"Final HPWL (absolute)             = {fhpwl:.2f}")
        print(f"Final Normalized Area             = {faN:.3f}")
        print(f"Final Normalized HPWL             = {fhN:.3f}")
        print(f"Final Normalized Penalty          = {fpN:.3f}")
        print(f"Final Cost (w={Global_w:.2f}, r=1.0)   = {final_cost:.3f}")
        print(f"DeadSpace Rate                    = {ds:.2f}%")

        # 최종 배치 플롯
        print("Plotting final optimized layout...")
        best_chip.plot_b_tree(iteration="Final_Optimized_SA")
        plt.show()
    else:
        print("Skipping Full SA.")

    print("\nFloorplanning process completed.")
