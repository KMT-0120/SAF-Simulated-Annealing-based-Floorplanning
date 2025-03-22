#해당 주석 문단은 지우지말것(특히 GPT 사용시, 지워지는지 또한 확인한 후 다시 추가), GPT 돌릴떄 함수 이름, 변수명 변경하지 말것
#3/8 iter 5000번마다 plot, SA종료 후 T 변화 추이 그래프 plot 추가
# 사용하지 않는 과거 코드 삭제(adjust_for_overlaps, check_overlap, is_within_bounds)
# plot관련해서 노드 크기/폰트 크기 조정, 노드 간격 조정
# c factor 조정

#3/9 GSRC 적용, 초기해 leftchain -> random으로 변경
#3/15 왼쪽 자식의 배치가 부모노드와 너무 조금 떨어져있던 버그 수정
#3/18 cost에 penalty추가, 정규화는 area_norm으로 해둠, 
#     area_norm이 HPWL_norm과 너무 크게 차이나는것을 확인하였기에(EX:4, 2000) area_norm * 500해서 사용
#     초기, 마지막 비용값을 더 잘 확인하기 위해 get_nomalized_values()함수 추가(cal_cost와 매커니즘 같으니 cal_cost 수정시에 같이 수정해줄것)
#     default_parent를 1000, 1000대신 전체 module의 넓이*1.2로 설정
#3/22 rotate, swap, move를 고를 때 랜덤으로 고르는 것이 아니라 sample move로 여러가지 moving 해보고 cost가 제일 낮은 걸 고르는 것처럼, 동작 세가지 모두 해보고 BESTlocalcost를 고름.
#     그리고 어떠한 상황에서는 어떠한 동작이 좋다는 것을 학습(Q-learning). 그래서 초기 온도를 랜덤하게 하는게 아니라 1e5, 1e6정도로 얼추 높게 설정 후 시작
#     Learn_to_Floorplan_through_Acquisition_of_Effective_Local_Search_Heuristics 참조했음.
#     + 초기배치를 랜덤이 아닌 SA로 수정. def partial_sa_for_initial()함수.
#     왜냐하면 초기 온도가 급격하게 내려오면, 정돈이 안된채로 좁은 local 에서만 변동을 줌. 
#     그래서 어느정도 정돈을 시키고 메인 함수에서 정돈된 block들을 높은 온도에서 탐색 시키면 좀 더 효율적이라고 생각했음. 
#     main에서의 c=20, sample_move = 30, 2구간 범위는 iter의 한 1/5정도 잡는게 좋은거 같음.

import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import random
import re

# ─────────────────────────────────────────────────────────────
# 1. Module, BTreeNode, Chip 클래스 (Random)
# ─────────────────────────────────────────────────────────────
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
        # PARENT 모듈 외의 모든 모듈
        self.modules = [m for m in modules if m.type != 'PARENT']

        # PARENT 모듈이 있는지 확인
        self.bound = next((m for m in modules if m.type == 'PARENT'), None)
        if not self.bound:
            total_area = sum(m.area for m in self.modules)
            side = math.sqrt(total_area * 1.2)
            self.bound = Module(
                name='DefaultParent',
                width=side,
                height=side,
                module_type='PARENT'
            )

        self.build_b_tree()

        self.contour_line = []
        self.max_width = 0
        self.max_height = 0

    def build_b_tree(self):
        if not self.modules:
            self.root = None
            return
    
        # 완전 왼쪽 체인 대신, 랜덤하게 parent를 골라 left/right에 붙이는 식으로
        random_mods = self.modules[:]
        random.shuffle(random_mods)
    
        self.root = BTreeNode(random_mods[0], parent=None)
        for mod in random_mods[1:]:
            node = BTreeNode(mod)
            cand_nodes = self.find_possible_parents()
            chosen_parent = random.choice(cand_nodes)
            self.attach_node(chosen_parent, node)

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

    # ─────────────────────────────────────────
    # B*-tree 연산들
    def rotate_node(self,node):
        node.module.rotate()

    def move_node(self,node):
        if node.parent is None:
            return f"[Op2: Move] Cannot move ROOT node ({node.module.name})."
        self.remove_node_from_parent(node)
        cands=self.find_possible_parents()
        if not cands:
            return "[Op2: Move] No available spot to re-insert node."
        new_par=random.choice(cands)
        side=self.attach_node(new_par,node)
        return f"[Op2: Move] Moved {node.module.name} under {new_par.module.name} ({side})"

    def remove_node_from_parent(self,node):
        p=node.parent
        if p is None:
            return
        if p.left==node:
            p.left=None
        elif p.right==node:
            p.right=None
        node.parent=None

    def find_possible_parents(self):
        all_nd=self.collect_all_nodes()
        results=[]
        for nd in all_nd:
            if nd.left is None or nd.right is None:
                results.append(nd)
        return results

    def attach_node(self,parent,node):
        node.parent=parent
        slots=[]
        if not parent.left:
            slots.append("left")
        if not parent.right:
            slots.append("right")
        if not slots:
            return "NoSlot"
        chosen=random.choice(slots)
        if chosen=="left":
            parent.left=node
        else:
            parent.right=node
        return chosen

    def swap_nodes(self,nodeA,nodeB):
        nodeA.module,nodeB.module=nodeB.module,nodeA.module

    def apply_specific_operation(self, op, all_nodes=None):
        """
        RL에서 액션(rotate/move/swap)이 결정된 경우
        """
        if all_nodes is None:
            all_nodes = self.collect_all_nodes()
            if not all_nodes:
                print("[Warn] No nodes in the tree.")
                return ("NoNode", None, None)

        if op=="rotate":
            chosen=random.choice(all_nodes)
            self.rotate_node(chosen)
            msg=f"[Op1: Rotate] Rotated {chosen.module.name}"
            op_data={'op':'rotate','nodeA_name': chosen.module.name}
        elif op=="move":
            chosen=random.choice(all_nodes)
            msg=self.move_node(chosen)
            op_data={'op':'move','nodeA_name': chosen.module.name}
        elif op=="swap":
            if len(all_nodes)<2:
                msg="[Op3: Swap] Not enough nodes to swap."
                op_data={'op':'swap','nodeA_name':None,'nodeB_name':None}
            else:
                nodeA,nodeB=random.sample(all_nodes,2)
                self.swap_nodes(nodeA,nodeB)
                msg=f"[Op3: Swap] Swapped {nodeA.module.name} <-> {nodeB.module.name}"
                op_data={
                    'op':'swap',
                    'nodeA_name': nodeA.module.name,
                    'nodeB_name': nodeB.module.name
                }
        else:
            msg="[InvalidOp]"
            op_data={'op':None}

        print(msg)
        # 좌표 재계산
        for nd in all_nodes:
            nd.module.x=0
            nd.module.y=0
            nd.module.order=None
        self.calculate_coordinates()
        return (msg, op, op_data)

    def reapply_operation(self, op_data):
        """
        sample 시 저장한 op_data대로 "동일한 노드/동일한 연산"을 다시 적용
        """
        if not op_data:
            return "[NoOpData]", "NoOp"
        all_nodes = self.collect_all_nodes()

        def find_node_by_name(nm):
            for nd in all_nodes:
                if nd.module.name == nm:
                    return nd
            return None

        op_type = op_data.get('op', 'NoOp')
        if op_type == 'rotate':
            nodeA_name = op_data.get('nodeA_name')
            ndA = find_node_by_name(nodeA_name)
            if ndA is None:
                return "[Rotate] Node not found", op_type
            self.rotate_node(ndA)
            msg = f"[Op1: Rotate] Re-applied rotate on {nodeA_name}"
        elif op_type == 'move':
            nodeA_name = op_data.get('nodeA_name')
            ndA = find_node_by_name(nodeA_name)
            if ndA is None:
                return "[Move] Node not found", op_type
            msg = self.move_node(ndA)
            msg += " [Re-applied]"
        elif op_type == 'swap':
            nodeA_name = op_data.get('nodeA_name')
            nodeB_name = op_data.get('nodeB_name')
            ndA = find_node_by_name(nodeA_name)
            ndB = find_node_by_name(nodeB_name)
            if not ndA or not ndB:
                return "[Swap] Node(s) not found", op_type
            self.swap_nodes(ndA, ndB)
            msg = f"[Op3: Swap] Re-applied swap: {nodeA_name} <-> {nodeB_name}"
        else:
            msg = "[NoOp or invalid op]"
            op_type = "NoOp"

        for nd in all_nodes:
            nd.module.x=0
            nd.module.y=0
            nd.module.order=None
        self.calculate_coordinates()

        return msg, op_type

    def randomize_b_tree(self,w=0.5):
        random.shuffle(self.modules)
        self.build_b_tree()
        self.calculate_coordinates()
        return calc_combined_cost(self.modules,w,chip=self)

    # ─────────────────────────────────────────
    # 좌표 배치 (Contour + DFS)
    def calculate_coordinates(self):
        if not self.root:
            return
        self.contour_line=[]
        self.max_width=0
        self.max_height=0
        all_nd=self.collect_all_nodes()
        for nd in all_nd:
            nd.module.x=0
            nd.module.y=0
            nd.module.order=None
        self._dfs_place_node(self.root,1,0,0)

    def _dfs_place_node(self,node,order,x_offset,y_offset):
        if not node:
            return order
        node.module.x=x_offset
        node.module.y=y_offset
        node.module.order=order

        x1=x_offset
        x2=x_offset+node.module.width
        top_y=y_offset+node.module.height

        self.insert_contour_segment(x1,x2,top_y)
        if x2>self.max_width:
            self.max_width=x2
        if top_y>self.max_height:
            self.max_height=top_y
        order+=1

        # 왼쪽 => x+width
        if node.left:
            lx=node.module.x+node.module.width
            rx=lx+node.left.module.width
            ly=self.update_contour(lx,rx)
            order=self._dfs_place_node(node.left,order,lx,ly)

        # 오른쪽 => x
        if node.right:
            rxs=node.module.x
            rxe=rxs+node.right.module.width
            rys=self.update_contour(rxs,rxe)
            order=self._dfs_place_node(node.right,order,rxs,rys)

        return order

    def update_contour(self,x1,x2):
        base_y=0
        for seg in self.contour_line:
            if not(seg.x2<=x1 or seg.x1>=x2):
                base_y=max(base_y,seg.y2)
        return base_y

    def insert_contour_segment(self,x1,x2,new_top):
        updated=[]
        i=0
        while i<len(self.contour_line):
            seg=self.contour_line[i]
            if seg.x2<=x1 or seg.x1>=x2:
                updated.append(seg)
            else:
                if seg.x1<x1 and seg.x2>x2:
                    updated.append(ContourNode(seg.x1,x1,seg.y2))
                    updated.append(ContourNode(x2,seg.x2,seg.y2))
                elif seg.x1<x1 and seg.x2<=x2:
                    updated.append(ContourNode(seg.x1,x1,seg.y2))
                elif seg.x1>=x1 and seg.x2>x2:
                    updated.append(ContourNode(x2,seg.x2,seg.y2))
            i+=1
        updated.append(ContourNode(x1,x2,new_top))
        updated.sort(key=lambda s:s.x1)
        self.contour_line=updated

    def plot_b_tree(self, iteration=None):
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(24,12))

        # (1) 물리적 배치
        self._plot_node(self.root,ax1)
        bound_rect=plt.Rectangle((0,0),self.bound.width,self.bound.height,
                                 edgecolor='red',facecolor='none',lw=2)
        ax1.add_patch(bound_rect)

        if iteration is not None:
            ax1.set_title(f"B*-Tree Physical Placement (Left-Chain, Iter={iteration})")
        else:
            ax1.set_title("B*-Tree Physical Placement (Left-Chain)")

        ax1.set_xlabel("X-coordinate")
        ax1.set_ylabel("Y-coordinate")
        ax1.set_xlim(0,self.bound.width+200)
        ax1.set_ylim(0,self.bound.height+200)

        # (2) B*-Tree 구조 (더 넓은 x간격, 작아진 노드/폰트)
        G=nx.DiGraph()
        pos={}
        self.add_edges(self.root,None,0,0,G,pos)

        nx.draw(G, pos, ax=ax2, with_labels=True, node_color="lightblue",
                edge_color="black", node_size=600, font_size=7)
        if iteration is not None:
            ax2.set_title(f"B*-Tree Structure (Left-Chain, Iter={iteration})")
        else:
            ax2.set_title("B*-Tree Structure (Left-Chain)")

    def _plot_node(self,node,ax):
        if not node:
            return
        m=node.module
        face_c='lightblue'
        rect=plt.Rectangle((m.x,m.y),m.width,m.height,
                           edgecolor='blue',facecolor=face_c,fill=True,lw=2)
        ax.add_patch(rect)
        ax.text(m.x+m.width/2,m.y+m.height/2,
                f'{m.name}\n#{m.order}',ha='center',va='center',fontsize=8)
        self._plot_node(node.left,ax)
        self._plot_node(node.right,ax)

    def add_edges(self,node,parent_name,depth,x_pos,G,pos):
        """
        노드 간 거리를 벌리기 위해
        - 수평은 x_pos*2
        - 수직은 -depth*2
        오른쪽 자식 지나갈 때 x_pos+2
        """
        if not node:
            return x_pos
        node_name=node.module.name
        pos[node_name]=(x_pos*2, -depth*2)

        G.add_node(node_name,label=node.module.name)
        if parent_name:
            G.add_edge(parent_name,node_name)

        x_pos=self.add_edges(node.left,node_name,depth+1,x_pos,G,pos)
        x_pos=self.add_edges(node.right,node_name,depth+1,x_pos+2,G,pos)
        return x_pos

# ─────────────────────────────────────────────────────────────
# 2. Cost 계산 함수들
# ─────────────────────────────────────────────────────────────

def calculate_hpwl(modules):
    net_dict={}
    for m in modules:
        for net_name in m.net:
            if net_name not in net_dict:
                net_dict[net_name]=[]
            net_dict[net_name].append(m)
    if not net_dict:
        return 0
    total_hpwl=0
    for net,mods in net_dict.items():
        min_x=min(mm.x for mm in mods)
        max_x=max(mm.x+m.width for mm in mods)
        min_y=min(mm.y for mm in mods)
        max_y=max(mm.y+m.height for mm in mods)
        net_hpwl=(max_x - min_x)+(max_y - min_y)
        total_hpwl+=net_hpwl
    return total_hpwl

def calculate_total_area(modules):
    if not modules:
        return 0,0,0
    min_x=min(m.x for m in modules)
    max_x=max(m.x+m.width for m in modules)
    min_y=min(m.y for m in modules)
    max_y=max(m.y+m.height for m in modules)
    width=(max_x - min_x)
    height=(max_y - min_y)
    return width,height,(width*height)

def calc_combined_cost(modules, w=0.5, chip=None, r=1.0):
    """
    cost = w * area_norm + (1-w) * hpwl_norm + r * penalty_norm
    """
    cost_scale=100
    if not modules:
        return 0.0

    base_scale=sum(m.area for m in modules)
    net_area_sum=sum(m.area for m in modules if m.net)
    if net_area_sum==0:
        net_area_sum=base_scale

    w_,h_,area_bb=calculate_total_area(modules)
    hpwl=calculate_hpwl(modules)

    if chip:
        bound_w = chip.bound.width
        bound_h = chip.bound.height
        penalty_sum = 0.0
        for mm in modules:
            over_w = (mm.x + mm.width) - bound_w
            over_h = (mm.y + mm.height) - bound_h
            if over_w > 0:
                penalty_sum += over_w**2
            if over_h > 0:
                penalty_sum += over_h**2
    else:
        penalty_sum = 0.0

    area_norm = (area_bb/base_scale) if base_scale else 0
    hpwl_norm = (hpwl/(2*math.sqrt(net_area_sum))) if net_area_sum>0 else 0
    penalty_norm = (penalty_sum/base_scale) if base_scale else 0

    area_norm *= 500  # 스케일 보정

    cost_value = w*area_norm + (1-w)*hpwl_norm + r*penalty_norm
    return cost_value * cost_scale

def get_normalized_values(modules, chip=None, w=0.5, r=1.0):
    if not modules:
        return (0.0, 0.0, 0.0)

    base_scale = sum(m.area for m in modules)
    net_area_sum = sum(m.area for m in modules if m.net)
    if net_area_sum==0:
        net_area_sum=base_scale

    _,_,area_bb = calculate_total_area(modules)
    hpwl        = calculate_hpwl(modules)

    if chip:
        bound_w = chip.bound.width
        bound_h = chip.bound.height
        penalty_sum = 0.0
        for mm in modules:
            over_w = (mm.x + mm.width) - bound_w
            over_h = (mm.y + mm.height) - bound_h
            if over_w > 0:
                penalty_sum += over_w**2
            if over_h > 0:
                penalty_sum += over_h**2
    else:
        penalty_sum = 0.0

    area_norm = (area_bb/base_scale) if base_scale else 0
    hpwl_norm = (hpwl/(2*math.sqrt(net_area_sum))) if net_area_sum>0 else 0
    penalty_norm = penalty_sum/base_scale if base_scale else 0

    area_norm *= 500
    return (area_norm, hpwl_norm, penalty_norm)

# ─────────────────────────────────────────────────────────────
# 3. FastSA 함수 + "연산 선정시 Q-Learning" (sample_moves)
#    => apply_random_operation() 제거, T1을 고정값으로.
# ─────────────────────────────────────────────────────────────

Q_TABLE = {}
ACTIONS = ["rotate","move","swap"]
epsilon = 0.3
alpha   = 0.5
gamma   = 0.9

def get_state_key(chip):
    cost_val = calc_combined_cost(chip.modules, w=0.5, chip=chip, r=1.0)
    discrete = round(cost_val, 1)  # 0.1단위로 반올림
    return discrete

def select_action_by_q(state_key):
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    else:
        if state_key not in Q_TABLE:
            Q_TABLE[state_key] = [0.0,0.0,0.0]
        arr = Q_TABLE[state_key]
        return arr.index(max(arr))

def update_q_value(s_key, a_idx, reward, s_next_key):
    if s_key not in Q_TABLE:
        Q_TABLE[s_key] = [0.0,0.0,0.0]
    if s_next_key not in Q_TABLE:
        Q_TABLE[s_next_key] = [0.0,0.0,0.0]
    q_old = Q_TABLE[s_key][a_idx]
    max_next = max(Q_TABLE[s_next_key])
    q_new = q_old + alpha*(reward + gamma*max_next - q_old)
    Q_TABLE[s_key][a_idx] = q_new

def fast_sa(chip, max_iter=5000, P=0.99, c=20, w=0.66, sample_moves=10, r=1.0):
    import copy, math, random

    # (A) T1을 임의의 큰 값으로 고정 (apply_random_operation 사용 안 함)
    T1 = 1e5
    print(f"Set T1={T1} (fixed)")

    best_chip = copy.deepcopy(chip)
    best_cost = calc_combined_cost(chip.modules, w, chip=chip, r=r)
    cur_cost  = best_cost

    temps = []
    for n in range(1, max_iter + 1):
        st_cp = copy.deepcopy(chip)
        oldc  = cur_cost

        cost_diffs = []
        best_local_cost = float('inf')
        best_op_data    = None
        best_msg        = None

        # (B) sample_moves번 반복 => RL로 연산 선택
        for _ in range(sample_moves):
            s_key = get_state_key(st_cp)
            a_idx = select_action_by_q(s_key)
            action_name = ACTIONS[a_idx]

            old_cost = calc_combined_cost(st_cp.modules, w, chip=st_cp, r=r)
            msg_tmp, op_tmp, op_data_tmp = st_cp.apply_specific_operation(action_name)
            new_cost = calc_combined_cost(st_cp.modules, w, chip=st_cp, r=r)

            reward = (old_cost - new_cost)
            s_next_key = get_state_key(st_cp)
            update_q_value(s_key, a_idx, reward, s_next_key)

            dE_ = new_cost - oldc
            cost_diffs.append(abs(dE_))
            if new_cost < best_local_cost:
                best_local_cost = new_cost
                best_op_data    = op_data_tmp
                best_msg        = msg_tmp

            # revert to original for next sample
            st_cp = copy.deepcopy(chip)

        # (C) sample_moves 결과로 delta_cost 계산 -> 온도 업데이트 (SA)
        if cost_diffs:
            delta_cost = max(sum(cost_diffs)/len(cost_diffs), 1e-6)
        else:
            delta_cost = 1e-6

        # T 업데이트 (기존 로직)
        if n == 1:
            T = T1
        elif 2 <= n <= 3000:
            T = max((T1 * delta_cost)/(n * c), 1e-6)
        else:
            T = max((T1 * delta_cost)/n, 1e-6)
        temps.append(T)

        # (D) best_op_data 실제 적용
        old_chip = copy.deepcopy(chip)
        old_cost = cur_cost
        re_msg, re_op = chip.reapply_operation(best_op_data)

        new_c = calc_combined_cost(chip.modules, w, chip=chip, r=r)
        dE = new_c - old_cost

        # (E) SA acceptance
        if dE < 0:
            cur_cost = new_c
            if new_c < best_cost:
                best_cost = new_c
                best_chip = copy.deepcopy(chip)
            acc_str  = "ACCEPT (better)"
            acc_prob = 1.0
        else:
            if T < 1e-12:
                acc_prob = 0.0
            else:
                acc_prob = math.exp(-abs(dE) / T)
            if random.random() < acc_prob:
                cur_cost = new_c
                if new_c < best_cost:
                    best_cost = new_c
                    best_chip = copy.deepcopy(chip)
                acc_str = "ACCEPT (worse)"
            else:
                chip = copy.deepcopy(old_chip)
                cur_cost = old_cost
                acc_str = "REJECT"

        print(f"[Iter={n:3d}] BestLocalOpData={best_op_data}, BestLocalCost={best_local_cost:.3f}, "
              f"ReAppMsg={re_msg}, T={T:9.5f}, dE={dE:9.5f}, Prob={acc_prob:6.4f}, {acc_str}")

        # plot every 40000 (사용자 요청대로 5000번마다 등으로 바꿔도 가능)
        if n % 5000 == 0:
            best_chip.plot_b_tree(iteration=n)
            plt.show()

    # (F) T vs Iter plot
    plt.figure()
    plt.plot(range(1, max_iter + 1), temps)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.title("Temperature vs. Iteration")
    plt.show()

    return best_chip


# ─────────────────────────────────────────────────────────────
# 4. 부분적인 SA(짧은 SA)를 먼저 돌려 초기 레이아웃을 개선하는 함수
# ─────────────────────────────────────────────────────────────
def partial_sa_for_initial(chip, pre_iter=500):
    """
    부분적으로 짧은 SA를 먼저 돌려서, 
    완전 무작위 초기 해보다는 어느 정도 정돈된 상태를 얻기 위한 함수.
    """
    print("\n[Info] --- Running partial SA for initial layout ---\n")
    # 예: iteration 200, sample_moves=2, c=5 등으로 
    # 매우 간단히 탐색만 해서 초기 해의 cost를 조금 낮춰 둔다.
    improved_chip = fast_sa(
        chip, 
        max_iter=pre_iter, 
        c=100, 
        w=0.66, 
        sample_moves=30, 
        r=1.0
    )
    print("[Info] --- Partial SA ended. Use this result as initial layout ---\n")
    return improved_chip

# ─────────────────────────────────────────────────────────────
# 5. 파일 파싱(Yal, GSRC)
# ─────────────────────────────────────────────────────────────
def parse_yal(file_path):
    modules=[]
    with open(file_path,'r') as f:
        lines=f.readlines()
    module_data={}
    in_network=False
    for line in lines:
        line=line.strip()
        if not line or line.startswith("/*") or line.startswith("*"):
            continue
        if line.startswith("MODULE"):
            module_name=line.split()[1].strip(';')
            module_data={'name':module_name,'net':[]}
            continue
        if line.startswith("TYPE"):
            module_type=line.split()[1].strip(';')
            module_data['type']=module_type
            continue
        if line.startswith("DIMENSIONS"):
            dims=list(map(float,line.replace("DIMENSIONS","").replace(";","").split()))
            x_coords=[dims[i] for i in range(0,len(dims),2)]
            y_coords=[dims[i+1] for i in range(0,len(dims),2)]
            module_data['width']=max(x_coords)-min(x_coords)
            module_data['height']=max(y_coords)-min(y_coords)
            continue
        if line.startswith("NETWORK"):
            in_network=True
            continue
        if line.startswith("ENDNETWORK"):
            in_network=False
            continue
        if line.startswith("ENDMODULE"):
            modules.append(Module(
                name=module_data['name'],
                width=module_data['width'],
                height=module_data['height'],
                module_type=module_data['type'],
                net=module_data['net']
            ))
            module_data={}
            continue
        if in_network:
            module_data['net'].append(line.strip(';'))
            continue
    return modules

def parse_gsrc_blocks(blocks_file):
    modules=[]
    pattern=re.compile(r"^(sb\d+)\s+\S+\s+(\d+)\s+(.*)$")
    with open(blocks_file,'r') as f:
        lines=f.readlines()

    for line in lines:
        line=line.strip()
        if not line or line.startswith('#') or line.startswith('UCSC'):
            continue
        if line.startswith('p') and 'terminal' in line:
            continue

        match=pattern.match(line)
        if match:
            blk_name=match.group(1)
            coords=match.group(3)
            cpat=re.compile(r"\(([\-\d\.]+)\s*,\s*([\-\d\.]+)\)")
            found = cpat.findall(coords)
            xs, ys=[],[]
            for (sx,sy) in found:
                xs.append(float(sx))
                ys.append(float(sy))
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            w_=maxx-minx
            h_=maxy-miny
            modules.append(Module(blk_name,w_,h_,'BLOCK',net=[]))

    return modules

def parse_gsrc_nets(nets_file, modules):
    name_map={m.name:m for m in modules}
    i=0
    with open(nets_file,'r') as f:
        lines=f.readlines()
    net_id=0
    while i<len(lines):
        line=lines[i].strip()
        if not line or line.startswith('#') or line.startswith('UCLA'):
            i+=1
            continue
        if line.startswith('NetDegree'):
            parts=line.split(':')
            deg_str=(parts[1].strip() if len(parts)>1 else "0")
            try:
                deg=int(deg_str)
            except:
                deg=0
            net_name=f"Net{net_id}"
            net_id+=1
            pins=[]
            for _ in range(deg):
                i+=1
                pin_ln=lines[i].strip()
                pin_prt=pin_ln.split()
                if pin_prt:
                    pin_name=pin_prt[0]
                    if pin_name in name_map:
                        pins.append(pin_name)
            for pn in pins:
                name_map[pn].net.append(net_name)
        i+=1
    return modules

# ─────────────────────────────────────────────────────────────
# 6. 메인 실행부 예시
# ─────────────────────────────────────────────────────────────
if __name__=="__main__":
    blocks_file = "./example/n100.blocks"
    nets_file   = "./example/n100.nets"
    modules = parse_gsrc_blocks(blocks_file)
    modules = parse_gsrc_nets(nets_file, modules)

    # [A] 초기 Chip 생성 -> 완전 무작위(랜덤) 배치
    chip = Chip(modules)
    chip.calculate_coordinates()
    chip.plot_b_tree()

    w_, h_, area_ = calculate_total_area(chip.modules)
    hpwl_ = calculate_hpwl(chip.modules)
    init_cost = calc_combined_cost(chip.modules, w=0.66, chip=chip, r=1.0)
    area_norm, hpwl_norm, penalty_norm = get_normalized_values(chip.modules, chip=chip, w=0.66, r=1.0)

    print("=== 초기 Chip (Random) ===")
    print(f"BoundingBox area (absolute)   = {area_}")
    print(f"HPWL (absolute)              = {hpwl_}")
    print(f"Area_norm                    = {area_norm:.3f}")
    print(f"HPWL_norm                    = {hpwl_norm:.3f}")
    print(f"Penalty_norm                 = {penalty_norm:.3f}")
    print(f"Initial cost (with penalty)  = {init_cost:.3f}")

    # [B] 부분적 SA(짧은 SA) -> 초기 배치 개선
    chip = partial_sa_for_initial(chip, pre_iter=500)  # 원하는 만큼 iteration 조정
    chip.plot_b_tree()
    plt.show()

    # [C] 메인 SA 실행 여부
    ans = input("FastSA + RL로 최적화?(y/n): ")
    if ans.lower().startswith('y'):
        best_chip = fast_sa(
            chip,
            max_iter=15000,  # 실제 큰 SA
            P=0.95,
            c=20,           # 여기서 c도 조정 가능
            w=0.66,
            sample_moves=30,
            r=1.0
        )
        final_w, final_h, final_area = calculate_total_area(best_chip.modules)
        final_hpwl = calculate_hpwl(best_chip.modules)
        final_cost = calc_combined_cost(best_chip.modules, w=0.5, chip=best_chip, r=1.0)
        f_area_norm, f_hpwl_norm, f_penalty_norm = get_normalized_values(best_chip.modules, best_chip, w=0.5, r=1.0)

        total_module_area = sum(m.area for m in best_chip.modules)
        if final_area>0:
            ds=((final_area - total_module_area)/final_area)*100
        else:
            ds=0.0

        print("=== FastSA+RL 종료 ===")
        print(f"최종 BoundingBox area (absolute)   = {final_area}")
        print(f"최종 HPWL (absolute)              = {final_hpwl}")
        print(f"최종 Area_norm                    = {f_area_norm:.3f}")
        print(f"최종 HPWL_norm                    = {f_hpwl_norm:.3f}")
        print(f"최종 Penalty_norm                 = {f_penalty_norm:.3f}")
        print(f"최종 cost (with penalty)          = {final_cost:.3f}")
        print(f"DeadSpaceRate                     = {ds:.3f}%")

        best_chip.plot_b_tree()
        plt.show()
    else:
        print("[Info] 종료")
