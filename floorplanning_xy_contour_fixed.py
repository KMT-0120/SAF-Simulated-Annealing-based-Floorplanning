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
#     dfs_place_node 함수 수정(오른쪽 자식 배치 더 붙을 수 있도록)


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


class VContourNode:
    """Vertical contour segment: [y1, y2) 구간에서 x2 우측 경계"""
    def __init__(self, y1, y2, x2):
        self.y1 = y1
        self.y2 = y2
        self.x2 = x2

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
            side = math.sqrt(total_area)
            self.bound = Module(
                name='DefaultParent',
                width=side,
                height=side,
                module_type='PARENT'
            )

        self.build_b_tree()
        self.contour_line = []
        self.v_contour = []  # vertical contour
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
        if node.parent is None:
            return f"[Op2: Move] Cannot move ROOT node ({node.module.name})."
        self.remove_node_from_parent(node)
        cands = self.find_possible_parents()
        if not cands:
            return "[Op2: Move] No available spot to re-insert node."
        new_par = random.choice(cands)
        side = self.attach_node(new_par,node)
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
        nodeA.module,nodeB.module = nodeB.module,nodeA.module

    def apply_specific_operation(self, op, all_nodes=None):
        """
        Q-learning등에서 rotate/move/swap이 결정된 뒤 실제 수행
        """
        if not all_nodes:
            all_nodes = self.collect_all_nodes()
        if not all_nodes:
            print("[Warn] No nodes in the tree.")
            return ("NoNode",None,None)

        if op=="rotate":
            chosen=random.choice(all_nodes)
            self.rotate_node(chosen)
            msg=f"[Op1: Rotate] Rotated {chosen.module.name}"
            op_data={'op':'rotate','nodeA_name':chosen.module.name}
        elif op=="move":
            chosen=random.choice(all_nodes)
            msg=self.move_node(chosen)
            op_data={'op':'move','nodeA_name':chosen.module.name}
        elif op=="swap":
            if len(all_nodes)<2:
                msg="[Op3: Swap] Not enough nodes to swap."
                op_data={'op':'swap','nodeA_name':None,'nodeB_name':None}
            else:
                nodeA,nodeB=random.sample(all_nodes,2)
                self.swap_nodes(nodeA,nodeB)
                msg=f"[Op3: Swap] Swapped {nodeA.module.name} <-> {nodeB.module.name}"
                op_data={'op':'swap','nodeA_name':nodeA.module.name,'nodeB_name':nodeB.module.name}
        else:
            msg="[InvalidOp]"
            op_data={'op':None}
        print(msg)

        for nd in all_nodes:
            nd.module.x=0
            nd.module.y=0
            nd.module.order=None
        self.calculate_coordinates()
        return (msg,op,op_data)

    def reapply_operation(self, op_data):
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
            self.rotate_node(ndA)
            msg=f"[Op1: Rotate] Re-applied rotate on {nA}"
        elif op_type=='move':
            nA=op_data.get('nodeA_name')
            ndA=find_node_by_name(nA)
            if ndA is None:
                return("[Move] Node not found",op_type)
            msg=self.move_node(ndA)
            msg+=" [Re-applied]"
        elif op_type=='swap':
            nA=op_data.get('nodeA_name')
            nB=op_data.get('nodeB_name')
            ndA=find_node_by_name(nA)
            ndB=find_node_by_name(nB)
            if (not ndA) or (not ndB):
                return("[Swap] Node(s) not found",op_type)
            self.swap_nodes(ndA,ndB)
            msg=f"[Op3: Swap] Re-applied swap: {nA} <-> {nB}"
        else:
            msg="[NoOp or invalid op]"
            op_type="NoOp"

        for nd in all_nd:
            nd.module.x=0
            nd.module.y=0
            nd.module.order=None
        self.calculate_coordinates()
        return(msg,op_type)

    def randomize_b_tree(self,w):
        random.shuffle(self.modules)
        self.build_b_tree()
        self.calculate_coordinates()
        return calc_combined_cost(self.modules,w,chip=self)

    def calculate_coordinates(self):
        if not self.root:
            return
        self.contour_line=[]
        self.v_contour=[]
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

        if node.left:
            lx=node.module.x+node.module.width
            rx=lx+node.left.module.width
            ly=self.update_contour(lx,rx)
            order=self._dfs_place_node(node.left,order,lx,ly)
        if node.right:
            rx_s=node.module.x
            rx_e=rx_s+node.right.module.width
            ry_s=self.update_contour(rx_s,rx_e)
            order=self._dfs_place_node(node.right,order,rx_s,ry_s)

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

    def update_v_contour(self, y1, y2):
        """주어진 y 구간 [y1,y2)에서 현 vertical contour 중 최대 x2 반환"""
        base_x = 0
        for seg in self.v_contour:
            if not (seg.y2 <= y1 or seg.y1 >= y2):  # overlap?
                base_x = max(base_x, seg.x2)
        return base_x

    def insert_v_contour_segment(self, y1, y2, new_right):
        """vertical contour에 새 세그먼트([y1,y2), x2=new_right) 삽입, 겹침 구간 분할"""
        updated = []
        i = 0
        while i < len(self.v_contour):
            seg = self.v_contour[i]
            if seg.y2 <= y1 or seg.y1 >= y2:
                updated.append(seg)
            else:
                if seg.y1 < y1 and seg.y2 > y2:
                    updated.append(VContourNode(seg.y1, y1, seg.x2))
                    updated.append(VContourNode(y2, seg.y2, seg.x2))
                elif seg.y1 < y1 and seg.y2 <= y2:
                    updated.append(VContourNode(seg.y1, y1, seg.x2))
                elif seg.y1 >= y1 and seg.y2 > y2:
                    updated.append(VContourNode(y2, seg.y2, seg.x2))
            i += 1
        updated.append(VContourNode(y1, y2, new_right))
        updated.sort(key=lambda s: s.y1)
        self.v_contour = updated


    def plot_b_tree(self, iteration=None):
            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(24,12))

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
            rect=plt.Rectangle((m.x,m.y),m.width,m.height,
                               edgecolor='blue',facecolor='lightblue',fill=True,lw=2)
            ax.add_patch(rect)
            ax.text(m.x+m.width/2,m.y+m.height/2,
                    f'{m.name}\n#{m.order}',ha='center',va='center',fontsize=8)
            self._plot_node(node.left,ax)
            self._plot_node(node.right,ax)

    def add_edges(self,node,parent_name,depth,x_pos,G,pos):
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
    # HPWL, 면적, penalty 함수들
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
        max_x=max(mm.x+mm.width for mm in mods)
        min_y=min(mm.y for mm in mods)
        max_y=max(mm.y+mm.height for mm in mods)
        net_hpwl=(max_x - min_x)+(max_y - min_y)
        total_hpwl+=net_hpwl
    return total_hpwl

def calculate_total_area(modules):
    if not modules:
        return (0,0,0)
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
    cost_scale=10
    if not modules:
        if return_all:
            return (0.0, 0.0, 0.0, 0.0)
        else:
            return 0.0

    base_scale=sum(m.area for m in modules)
    if base_scale<1e-12:
        base_scale=1.0
    net_area_sum=sum(m.area for m in modules if m.net)
    if net_area_sum<1e-12:
        net_area_sum=base_scale

    w_,h_,area_bb=calculate_total_area(modules)
    hpwl=calculate_hpwl(modules)

    # penalty
    if chip:
        bound_w = chip.bound.width
        bound_h = chip.bound.height
        w_bb, h_bb, _ = calculate_total_area(modules)

        if w_bb<=bound_w and h_bb<=bound_h:
            area_violation=0.0
        elif w_bb>bound_w and h_bb<=bound_h:
            area_violation=(w_bb-bound_w)*bound_h
        elif w_bb<=bound_w and h_bb>bound_h:
            area_violation=(h_bb-bound_h)*bound_w
        else:
            area_violation=(w_bb*h_bb)-(bound_w*bound_h)

        length_violation=0.0
        for mm in modules:
            over_w=(mm.x+mm.width)-bound_w
            over_h=(mm.y+mm.height)-bound_h
            if over_w>0:
                length_violation+=(over_w**2)
            if over_h>0:
                length_violation+=(over_h**2)

        penalty_sum=area_violation+length_violation
    else:
        penalty_sum=0.0

    area_norm   = area_bb/base_scale
    hpwl_norm   = hpwl/(2*math.sqrt(net_area_sum))
    penalty_norm= penalty_sum/base_scale

    area_norm   *=500
    penalty_norm*=50

    cost_val= w*area_norm + (1-w)*hpwl_norm + r*penalty_norm
    cost_val*= cost_scale

    if return_all:
        return (cost_val, area_norm, hpwl_norm, penalty_norm)
    else:
        return cost_val

# ─────────────────────────────────────────────────────────────
# Q-Learning / fast_sa
# ─────────────────────────────────────────────────────────────

Q_TABLE={}
ACTIONS=["rotate","move","swap"]
epsilon=0.3
alpha=0.5
gamma=0.9

def get_state_key(chip):
    c_val=calc_combined_cost(chip.modules, w=Global_w, chip=chip, r=1.0, return_all=False)
    return round(c_val,1)

def select_action_by_q(state_key):
    if random.random()<epsilon:
        return random.randrange(len(ACTIONS))
    else:
        if state_key not in Q_TABLE:
            Q_TABLE[state_key]=[0.0,0.0,0.0]
        arr=Q_TABLE[state_key]
        return arr.index(max(arr))

def update_q_value(s_key,a_idx,reward,s_next_key):
    if s_key not in Q_TABLE:
        Q_TABLE[s_key]=[0.0,0.0,0.0]
    if s_next_key not in Q_TABLE:
        Q_TABLE[s_next_key]=[0.0,0.0,0.0]
    q_old=Q_TABLE[s_key][a_idx]
    max_next=max(Q_TABLE[s_next_key])
    q_new=q_old+alpha*(reward+gamma*max_next-q_old)
    Q_TABLE[s_key][a_idx]=q_new

def fast_sa(chip,max_iter=5000,P=0.99,c=20,w=Global_w,sample_moves=10,r=1.0):
    import copy,math,random

    # (A) T1 추정
    orig_st=copy.deepcopy(chip)
    orig_cost=calc_combined_cost(chip.modules,w,chip=chip,r=r,return_all=False)
    up_diffs=[]
    for _ in range(sample_moves+10):
        s_key=get_state_key(chip)
        a_idx=select_action_by_q(s_key)
        action_name=ACTIONS[a_idx]

        old_c=calc_combined_cost(chip.modules,w,chip=chip,r=r,return_all=False)
        msg_tmp,op_tmp,op_data_tmp=chip.apply_specific_operation(action_name)
        new_c=calc_combined_cost(chip.modules,w,chip=chip,r=r,return_all=False)
        dE=new_c-old_c

        rew=(old_c-new_c)
        s_next=get_state_key(chip)
        update_q_value(s_key,a_idx,rew,s_next)

        if dE>0:
            up_diffs.append(dE)
        chip=copy.deepcopy(orig_st)

    if up_diffs:
        delta_avg=sum(up_diffs)/len(up_diffs)
    else:
        delta_avg=1.0
    if delta_avg<1e-12:
        delta_avg=1.0

    T1_scale_factor=1.0
    T1=abs(delta_avg/math.log(P))*T1_scale_factor
    print(f"Initial T1={T1:.3f}")

    best_chip=copy.deepcopy(chip)
    best_cost=orig_cost
    cur_cost =orig_cost

    temps=[]
    for n in range(1,max_iter+1):
        st_cp=copy.deepcopy(chip)
        oldc=cur_cost

        cost_diffs=[]
        best_local_cost=float('inf')
        best_op_data=None

        for _ in range(sample_moves):
            s_key2=get_state_key(st_cp)
            a_idx2=select_action_by_q(s_key2)
            act2=ACTIONS[a_idx2]

            old_c2=calc_combined_cost(st_cp.modules,w,chip=st_cp,r=r,return_all=False)
            msg2,op2,opd2=st_cp.apply_specific_operation(act2)
            new_c2=calc_combined_cost(st_cp.modules,w,chip=st_cp,r=r,return_all=False)

            rew2=(old_c2-new_c2)
            s_next2=get_state_key(st_cp)
            update_q_value(s_key2,a_idx2,rew2,s_next2)

            dE2=new_c2-oldc
            cost_diffs.append(abs(dE2))
            if new_c2<best_local_cost:
                best_local_cost=new_c2
                best_op_data=opd2

            st_cp=copy.deepcopy(chip)

        if cost_diffs:
            delta_cost=max(sum(cost_diffs)/len(cost_diffs),1e-6)
        else:
            delta_cost=1e-6

        if n==1:
            T=T1
        elif 2<=n<=300:
            T=max((T1*delta_cost)/(n*c),1e-6)
        else:
            T=max((T1*delta_cost)/n,1e-6)
        temps.append(T)

        old_chip=copy.deepcopy(chip)
        old_cost=cur_cost
        re_msg,re_op=chip.reapply_operation(best_op_data)

        new_c=calc_combined_cost(chip.modules,w,chip=chip,r=r,return_all=False)
        dE=new_c-old_cost

        if dE<0:
            cur_cost=new_c
            if new_c<best_cost:
                best_cost=new_c
                best_chip=copy.deepcopy(chip)
            acc_str="ACCEPT (better)"
            acc_prob=1.0
        else:
            if T<1e-12:
                acc_prob=0.0
            else:
                acc_prob=math.exp(-abs(dE)/T)
            if random.random()<acc_prob:
                cur_cost=new_c
                if new_c<best_cost:
                    best_cost=new_c
                    best_chip=copy.deepcopy(chip)
                acc_str="ACCEPT (worse)"
            else:
                chip=copy.deepcopy(old_chip)
                cur_cost=old_cost
                acc_str="REJECT"

        print(f"[Iter={n:3d}] BestLocalCost={best_local_cost:.3f}, ReAppMsg={re_msg}")
        print(f"T={T:9.5f}, dE={dE:9.5f}, Prob={acc_prob:6.4f}, {acc_str}")

        #if n%2000==0:
        #    best_chip.plot_b_tree(iteration=n)
        #    plt.show()

    plt.figure() 
    plt.plot(range(1,max_iter+1),temps)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.title("Temperature vs. Iteration")
    plt.show()

    return best_chip

def partial_sa_for_initial(chip, pre_iter=500):
    """
    부분적으로 짧은 SA를 먼저 돌려서,
    완전 무작위 초기 해보다는 어느 정도 정돈된 상태를 얻기 위한 함수.
    """
    print("\n[Info] --- Running partial SA for initial layout ---\n")
    improved_chip = fast_sa(
        chip,
        max_iter=pre_iter,
        c=100,
        w=Global_w,
        sample_moves=30,
        r=1.0
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
            found=cpat.findall(coords)
            xs,ys=[],[]
            for (sx,sy) in found:
                xs.append(float(sx))
                ys.append(float(sy))
            minx,maxx=min(xs),max(xs)
            miny,maxy=min(ys),max(ys)
            w_=maxx-minx
            h_=maxy-miny
            modules.append(Module(blk_name,w_,h_,'BLOCK',net=[]))
    return modules

def parse_gsrc_nets(nets_file,modules):
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
# 5. 메인 실행
# ─────────────────────────────────────────────────────────────
if __name__=="__main__":
    #blocks_file="./example/ami49.yal"
    #modules=parse_yal(blocks_file)
    blocks_file="./example/n300.blocks"
    nets_file  ="./example/n300.nets"
    modules=parse_gsrc_blocks(blocks_file)
    modules=parse_gsrc_nets(nets_file,modules)

    # [A] 초기 Chip 생성 -> 완전 무작위(랜덤) 배치
    chip=Chip(modules)
    chip.calculate_coordinates()
    chip.plot_b_tree()

    init_cost, aN,hN,pN = calc_combined_cost(chip.modules, w=Global_w, chip=chip, r=1.0, return_all=True)
    w_,h_,area_=calculate_total_area(chip.modules)
    hpwl_      =calculate_hpwl(chip.modules)
    print("=== 초기 Chip (Left-Chain) ===")
    print(f"BoundingBox area (absolute)   = {area_}")
    print(f"HPWL (absolute)              = {hpwl_}")
    print(f"Area_norm                    = {aN:.3f}")
    print(f"HPWL_norm                    = {hN:.3f}")
    print(f"Penalty_norm                 = {pN:.3f}")
    print(f"Initial cost (with penalty)  = {init_cost:.3f}")

    # [B] 부분적 SA(짧은 SA) -> 초기 배치 개선
    chip = partial_sa_for_initial(chip, pre_iter=500)  # 원하는 만큼 iteration 조정
    chip.plot_b_tree()
    plt.show()
    

    # [C] 메인 SA 실행 여부
    ans=input("FastSA(Q-Learning)로 최적화를 진행?(y/n): ")
    if ans.lower().startswith('y'):
        best_chip=fast_sa(chip,
                          max_iter=8000,
                          P=0.95,
                          c=100,
                          w=Global_w,
                          sample_moves=40,
                          r=1.0)

        final_cost, faN, fhN, fpN = calc_combined_cost(best_chip.modules, w=Global_w, chip=best_chip, r=1.0, return_all=True)
        fw_,fh_,farea_=calculate_total_area(best_chip.modules)
        fhpwl_=calculate_hpwl(best_chip.modules)

        total_mod_area=sum(m.area for m in best_chip.modules)
        if farea_>0:
            ds=((farea_-total_mod_area)/farea_)*100
        else:
            ds=0.0

        print("=== FastSA 종료 ===")
        print(f"최종 BoundingBox area (absolute)   = {farea_}")
        print(f"최종 HPWL (absolute)              = {fhpwl_}")
        print(f"최종 Area_norm                    = {faN:.3f}")
        print(f"최종 HPWL_norm                    = {fhN:.3f}")
        print(f"최종 Penalty_norm                 = {fpN:.3f}")
        print(f"최종 cost (with penalty)          = {final_cost:.3f}")
        print(f"DeadSpaceRate                     = {ds:.3f}%")

        best_chip.plot_b_tree()
        plt.show()