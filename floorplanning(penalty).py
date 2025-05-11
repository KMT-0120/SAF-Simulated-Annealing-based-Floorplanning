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
from matplotlib import font_manager, rc

sys.setrecursionlimit(10000)

# Matplotlib 한글 폰트 설정
try:
    font_path = None
    font_names_to_try = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Droid Sans Fallback']
    for name in font_names_to_try:
        try:
            font_check_path = font_manager.findfont(name, fallback_to_default=False)
            if font_check_path:
                rc('font', family=name)
                font_path = name
                break
        except: 
            pass

    if not font_path and any(name in f.name for f in font_manager.fontManager.ttflist for name in font_names_to_try):
         for name in font_names_to_try:
            if any(name in f.name for f in font_manager.fontManager.ttflist):
                rc('font', family=name)
                font_path = name
                break

    if not font_path and sys.platform == "darwin": 
         rc('font', family='AppleGothic')
         font_path = 'AppleGothic'
    elif not font_path and sys.platform.startswith("win"): 
         rc('font', family='Malgun Gothic')
         font_path = 'Malgun Gothic'
    
    if not font_path: 
        rc('font', family='sans-serif')

    plt.rcParams['axes.unicode_minus'] = False 
    if font_path :
        print(f"[Info] Matplotlib 한글 폰트 '{font_path}'로 설정됨.")
    else:
        print("[Warn] 적절한 한글 폰트를 찾지 못했습니다. 플롯의 한글이 깨질 수 있습니다.")
except Exception as e:
    print(f"[Warn] 한글 폰트 설정 중 오류 발생: {e}. 플롯의 한글이 깨질 수 있습니다.")


global Global_w
Global_w=0.66 
global Global_r_penalty # 1차 SA 및 기본 페널티 가중치
Global_r_penalty = 70.0
global Global_r_dead_space # 2차 SA에서 Dead Space에 대한 추가 가중치
Global_r_dead_space =  80 # 예시 값, 실험을 통해 조정 필요


class Module:
    def __init__(self, name: str, width: float, height: float, module_type: str, net=None):
        self.name = name
        self.width = width
        self.height = height
        self.area = width * height
        self.x = 0.0 # 부동소수점 사용 명시
        self.y = 0.0 # 부동소수점 사용 명시
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

        self.bound = None 
        # self.bound = next((m for m in modules if m.type == 'PARENT'), None) 

        if not self.bound:
            total_area = sum(m.area for m in self.modules)
            side = math.sqrt(total_area*1.2) 
            self.bound = Module(
                name='DefaultParent',
                width=side, 
                height=side, 
                module_type='PARENT'
            )

        self.root = None 
        self.build_b_tree() 
        self.contour_line = [] 
        self.max_width = 0.0 # 부동소수점 사용 명시
        self.max_height = 0.0 # 부동소수점 사용 명시


    def build_b_tree(self): 
        if not self.modules:
            self.root = None
            return

        all_current_btnodes = self.collect_all_nodes() 
        for btnode in all_current_btnodes:
            btnode.parent = None
            btnode.left = None
            btnode.right = None

        random_mods = self.modules[:] 
        random.shuffle(random_mods)

        self.root = BTreeNode(random_mods[0], parent=None) 
        
        nodes_to_insert = [BTreeNode(mod) for mod in random_mods[1:]]

        for node_to_be_inserted in nodes_to_insert: 
            cands = self._find_possible_parents_for_random_build() 
            if not cands:
                if self.root and (not self.root.left or not self.root.right):
                    cands = [self.root]
                else:
                    print(f"[Warning] build_b_tree: {node_to_be_inserted.module.name}을(를) 붙일 후보 부모 없음. 트리가 불완전할 수 있음.")
                    continue

            chosen_parent_btnode = random.choice(cands) 
            self.attach_node(chosen_parent_btnode, node_to_be_inserted)


    def _find_possible_parents_for_random_build(self):
        """랜덤 B-Tree 생성을 위해 현재 트리에서 빈 슬롯이 있는 BTreeNode들을 찾습니다."""
        if not self.root:
            return []
        
        possible_parents = [] 
        queue = [self.root]
        visited = {self.root} 

        while queue:
            current_btnode = queue.pop(0) 
            if not current_btnode.left or not current_btnode.right: 
                possible_parents.append(current_btnode)
            
            if current_btnode.left and current_btnode.left not in visited:
                queue.append(current_btnode.left)
                visited.add(current_btnode.left)
            if current_btnode.right and current_btnode.right not in visited:
                queue.append(current_btnode.right)
                visited.add(current_btnode.right)
        return possible_parents

    def collect_all_nodes(self): 
        if not self.root:
            return []
        result=[] 
        queue=[self.root]
        visited = {self.root} 
        while queue:
            cur_btnode=queue.pop(0) 
            result.append(cur_btnode)
            if cur_btnode.left and cur_btnode.left not in visited:
                queue.append(cur_btnode.left)
                visited.add(cur_btnode.left)
            if cur_btnode.right and cur_btnode.right not in visited:
                queue.append(cur_btnode.right)
                visited.add(cur_btnode.right)
        return result

    def rotate_node(self,btnode): 
        btnode.module.rotate()

    def move_node(self,btnode_to_move): 
        if btnode_to_move.parent is None: 
            return f"[Op2: Move] Cannot move ROOT node ({btnode_to_move.module.name})."
        
        original_parent_btnode = btnode_to_move.parent
        was_left_child = (original_parent_btnode.left == btnode_to_move)

        self.remove_node_from_parent(btnode_to_move) 
        
        candidate_parent_btnodes = self._find_possible_parents_for_random_build()
        
        if not candidate_parent_btnodes: 
            if original_parent_btnode and (not original_parent_btnode.left or not original_parent_btnode.right):
                if was_left_child and not original_parent_btnode.left:
                    original_parent_btnode.left = btnode_to_move
                    btnode_to_move.parent = original_parent_btnode
                elif not was_left_child and not original_parent_btnode.right: 
                    original_parent_btnode.right = btnode_to_move
                    btnode_to_move.parent = original_parent_btnode
            return "[Op2: Move] No available spot to re-insert node."

        new_parent_btnode = random.choice(candidate_parent_btnodes) 
        side = self.attach_node(new_parent_btnode, btnode_to_move)
        return f"[Op2: Move] Moved {btnode_to_move.module.name} under {new_parent_btnode.module.name} ({side})"


    def remove_node_from_parent(self,btnode): 
        parent_btnode = btnode.parent 
        if parent_btnode is None: 
            return
        if parent_btnode.left == btnode:
            parent_btnode.left = None
        elif parent_btnode.right == btnode:
            parent_btnode.right = None
        btnode.parent = None 

    def find_possible_parents(self): 
        all_btnodes = self.collect_all_nodes() 
        results=[]
        for btnode in all_btnodes:
            if btnode.left is None or btnode.right is None: 
                results.append(btnode)
        return results

    def attach_node(self,parent_btnode, btnode_to_attach): 
        btnode_to_attach.parent = parent_btnode 
        slots=[]
        if not parent_btnode.left:
            slots.append("left")
        if not parent_btnode.right:
            slots.append("right")
        
        if not slots: 
            print(f"[Warning] attach_node: Parent {parent_btnode.module.name} has no empty slots for {btnode_to_attach.module.name}.")
            return "NoSlot" 
            
        chosen_side=random.choice(slots)
        if chosen_side=="left":
            parent_btnode.left = btnode_to_attach
        else: 
            parent_btnode.right = btnode_to_attach
        return chosen_side 

    def swap_nodes(self,btnodeA, btnodeB): 
        btnodeA.module, btnodeB.module = btnodeB.module, btnodeA.module

    def apply_specific_operation(self, op, all_btnodes=None):
        if not all_btnodes:
            all_btnodes = self.collect_all_nodes() 
        if not all_btnodes:
            return ("NoNode",None,None) 

        op_data = {'op': op} 
        msg = "[InvalidOp]" 

        if op=="rotate":
            chosen_btnode = random.choice(all_btnodes)
            self.rotate_node(chosen_btnode)
            msg=f"[Op1: Rotate] Rotated {chosen_btnode.module.name}" 
            op_data['nodeA_name']=chosen_btnode.module.name
        elif op=="move":
            if len(all_btnodes) <= 1: 
                msg = "[Op2: Move] Not enough nodes to perform a move." 
                op_data['nodeA_name'] = None
            else:
                non_root_btnodes = [n for n in all_btnodes if n.parent is not None]
                if not non_root_btnodes: 
                    chosen_btnode = all_btnodes[0] 
                else:
                    chosen_btnode = random.choice(non_root_btnodes)
                
                msg=self.move_node(chosen_btnode) 
                op_data['nodeA_name']=chosen_btnode.module.name 
        elif op=="swap":
            if len(all_btnodes)<2:
                msg="[Op3: Swap] Not enough nodes to swap." 
                op_data['nodeA_name']=None
                op_data['nodeB_name']=None 
            else:
                btnodeA, btnodeB = random.sample(all_btnodes,2)
                self.swap_nodes(btnodeA, btnodeB)
                msg=f"[Op3: Swap] Swapped {btnodeA.module.name} <-> {btnodeB.module.name}" 
                op_data['nodeA_name']=btnodeA.module.name
                op_data['nodeB_name']=btnodeB.module.name
        else: 
            op_data['op'] = None 
        
        self.calculate_coordinates() 
        return (msg,op,op_data) 

    def reapply_operation(self, op_data):
        if not op_data or 'op' not in op_data:
            return ("[NoOpData or Invalid OpData]", "NoOp")

        all_btnodes=self.collect_all_nodes() 
        def find_btnode_by_module_name(name_str): 
            if name_str is None: return None
            for btnode_item in all_btnodes: 
                if btnode_item.module.name == name_str:
                    return btnode_item
            return None

        op_type=op_data.get('op','NoOp')
        msg = f"[{op_type}] 연산 재적용." 

        if op_type=='rotate':
            nodeA_name=op_data.get('nodeA_name')
            btnodeA=find_btnode_by_module_name(nodeA_name)
            if btnodeA is None:
                return(f"[{op_type}] Node {nodeA_name} not found",op_type) 
            self.rotate_node(btnodeA) 
            msg=f"[Op1: Rotate] Re-applied rotate on {nodeA_name}" 
        elif op_type=='move':
            nodeA_name=op_data.get('nodeA_name')
            btnodeA=find_btnode_by_module_name(nodeA_name)
            if btnodeA is None:
                return(f"[{op_type}] Node {nodeA_name} not found",op_type)
            if btnodeA.parent is None: 
                 msg = f"[Op2: Move] Cannot re-apply move to ROOT node {nodeA_name}." 
            else:
                msg=self.move_node(btnodeA)
                msg+=f" [Re-applied on {nodeA_name}]" 
        elif op_type=='swap':
            nodeA_name=op_data.get('nodeA_name')
            nodeB_name=op_data.get('nodeB_name')
            btnodeA=find_btnode_by_module_name(nodeA_name)
            btnodeB=find_btnode_by_module_name(nodeB_name)
            if (not btnodeA) or (not btnodeB):
                missing_nodes_str = []
                if not btnodeA: missing_nodes_str.append(nodeA_name)
                if not btnodeB: missing_nodes_str.append(nodeB_name)
                return(f"[{op_type}] Node(s) {', '.join(missing_nodes_str)} not found",op_type)
            self.swap_nodes(btnodeA, btnodeB) 
            msg=f"[Op3: Swap] Re-applied swap: {nodeA_name} <-> {nodeB_name}" 
        else:
            msg=f"[NoOp or invalid op_type: {op_type}]" 
            op_type="NoOp"
        
        self.calculate_coordinates() 
        return(msg,op_type) 

    def randomize_b_tree(self,w):
        """B*-Tree 구조를 랜덤하게 재구성하고 비용을 반환"""
        self.build_b_tree() 
        self.calculate_coordinates() 
        return calc_combined_cost(self.modules,w,chip=self, r_penalty=Global_r_penalty) # 기본 페널티 비용 사용

    def calculate_coordinates(self):
        if not self.root:
            self.max_width = 0.0
            self.max_height = 0.0
            self.contour_line = []
            for m_obj in self.modules: 
                m_obj.x = 0.0
                m_obj.y = 0.0
                m_obj.order = None
            return

        self.contour_line=[] 
        self.max_width=0.0
        self.max_height=0.0
        
        for mod_obj in self.modules:
            mod_obj.x = 0.0
            mod_obj.y = 0.0
            mod_obj.order = None
            
        self._dfs_place_node(self.root,1,0.0,0.0) 
    
    def _dfs_place_node(self,btnode,current_order,x_offset,y_offset): 
        if not btnode:
            return current_order

        current_module = btnode.module
        current_module.x = x_offset
        current_module.y = y_offset
        current_module.order = current_order 

        x1_current = x_offset
        x2_current = x_offset + current_module.width
        top_y_current = y_offset + current_module.height

        self.insert_contour_segment(x1_current, x2_current, top_y_current)
        if x2_current > self.max_width:
            self.max_width = x2_current
        if top_y_current > self.max_height:
            self.max_height = top_y_current
        
        order_after_current_node = current_order + 1

        if btnode.left:
            left_child_module = btnode.left.module
            lx_left_child = current_module.x + current_module.width 
            rx_left_child = lx_left_child + left_child_module.width 
            ly_left_child = self.update_contour(lx_left_child, rx_left_child) 
            order_after_current_node = self._dfs_place_node(btnode.left, order_after_current_node, lx_left_child, ly_left_child)
        
        if btnode.right:
            right_child_module = btnode.right.module
            rx_s_right_child = current_module.x 
            rx_e_right_child = rx_s_right_child + right_child_module.width 
            ry_s_right_child = self.update_contour(rx_s_right_child, rx_e_right_child) 
            order_after_current_node = self._dfs_place_node(btnode.right, order_after_current_node, rx_s_right_child, ry_s_right_child)

        return order_after_current_node

    def update_contour(self,x1_new,x2_new):
        base_y=0.0
        epsilon = 1e-9 
        for seg in self.contour_line:
            if not(seg.x2 <= x1_new + epsilon or seg.x1 >= x2_new - epsilon): 
                base_y=max(base_y,seg.y2)
        return base_y

    def insert_contour_segment(self,x1_new,x2_new,new_top_y):
        new_contour_list = []
        epsilon = 1e-9

        event_x_coords = {x1_new, x2_new}
        for seg_old in self.contour_line:
            event_x_coords.add(seg_old.x1)
            event_x_coords.add(seg_old.x2)
        
        sorted_unique_x = sorted(list(filter(lambda x: x is not None, event_x_coords)))

        if not sorted_unique_x:
            self.contour_line = [ContourNode(x1_new, x2_new, new_top_y)] 
            return

        merged_x_coords = [sorted_unique_x[0]]
        for i in range(1, len(sorted_unique_x)):
            if sorted_unique_x[i] > merged_x_coords[-1] + epsilon:
                merged_x_coords.append(sorted_unique_x[i])
        sorted_unique_x = merged_x_coords


        for i in range(len(sorted_unique_x) - 1):
            current_interval_x1 = sorted_unique_x[i]
            current_interval_x2 = sorted_unique_x[i+1]

            if current_interval_x2 <= current_interval_x1 + epsilon: 
                continue

            mid_point_x = (current_interval_x1 + current_interval_x2) / 2.0
            max_y_for_interval = 0.0

            for seg_old in self.contour_line:
                if seg_old.x1 <= mid_point_x + epsilon and seg_old.x2 >= mid_point_x - epsilon:
                    max_y_for_interval = max(max_y_for_interval, seg_old.y2)
            
            if x1_new <= mid_point_x + epsilon and x2_new >= mid_point_x - epsilon:
                max_y_for_interval = max(max_y_for_interval, new_top_y)
            
            if not new_contour_list or abs(new_contour_list[-1].y2 - max_y_for_interval) > epsilon or \
               abs(new_contour_list[-1].x2 - current_interval_x1) > epsilon :
                if current_interval_x2 > current_interval_x1 + epsilon: 
                    new_contour_list.append(ContourNode(current_interval_x1, current_interval_x2, max_y_for_interval))
            else: 
                new_contour_list[-1].x2 = current_interval_x2
        
        self.contour_line = new_contour_list


    def _check_rect_overlap(self, r1_x, r1_y, r1_w, r1_h, r2_x, r2_y, r2_w, r2_h):
        epsilon = 1e-9 
        if (r1_x < r2_x + r2_w - epsilon and
            r1_x + r1_w > r2_x + epsilon and
            r1_y < r2_y + r2_h - epsilon and
            r1_y + r1_h > r2_y + epsilon):
            return True
        return False

    def compact_floorplan_final(self):
        print("\n[Info] --- 최종 Greedy Compaction 수행 (왼쪽 -> 아래쪽) ---")
        if not self.modules:
            return
        
        modules_to_compact = self.modules[:] 

        modules_to_compact.sort(key=lambda m: (m.x, m.y)) 
        
        for i in range(len(modules_to_compact)):
            module_i = modules_to_compact[i]
            target_x_i = 0.0 
            for j in range(i): 
                module_j = modules_to_compact[j] 
                
                y_intervals_overlap = (module_i.y < module_j.y + module_j.height - 1e-9 and 
                                       module_i.y + module_i.height > module_j.y + 1e-9)

                if y_intervals_overlap:
                    target_x_i = max(target_x_i, module_j.x + module_j.width)
            
            module_i.x = target_x_i 

        modules_to_compact.sort(key=lambda m: (m.y, m.x)) 
        
        for i in range(len(modules_to_compact)):
            module_i = modules_to_compact[i]
            target_y_i = 0.0
            for j in range(i):
                module_j = modules_to_compact[j] 

                x_intervals_overlap = (module_i.x < module_j.x + module_j.width - 1e-9 and 
                                       module_i.x + module_i.width > module_j.x + 1e-9)

                if x_intervals_overlap:
                    target_y_i = max(target_y_i, module_j.y + module_j.height)
            
            module_i.y = target_y_i 

        if self.modules: 
            self.max_width = max(m.x + m.width for m in self.modules) if self.modules else 0.0
            self.max_height = max(m.y + m.height for m in self.modules) if self.modules else 0.0
        else:
            self.max_width = 0.0
            self.max_height = 0.0
        
        print("[Info] --- 최종 Greedy Compaction 완료 ---")

    def plot_b_tree(self, iteration=None, title_suffix=""): 
        fig_width = 20 
        fig_height = 10 
        if iteration and "Initial_Random" in iteration : 
            fig_width = 24 
            fig_height = 12
        
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(fig_width,fig_height)) 

        plot_title_p = "물리적 배치" 
        if iteration is not None:
            plot_title_p = f"물리적 배치 (Iter={iteration}{title_suffix})"
        
        for m_obj in self.modules: 
            edge_c = 'blue'
            face_c = 'lightblue'
            text_suffix = f'\n#{m_obj.order if m_obj.order is not None else "N/A"}' 
            
            if iteration: 
                iter_str_lower = str(iteration).lower()
                if "compacted" in iter_str_lower : # B-Tree 재구성 로직을 사용하지 않으므로, 이 조건만으로 충분
                    edge_c = 'green'
                    face_c = 'lightgreen'
                    text_suffix = '\n(Compacted)'
                # elif "reconstructed_btree" in iter_str_lower: # 이 부분은 B-Tree 재구성 안하므로 제거
                #     edge_c = 'purple'
                #     face_c = 'lavender'
                #     text_suffix = f'\n(ReconTree #{m_obj.order if m_obj.order is not None else "N/A"})'

            rect=plt.Rectangle((m_obj.x,m_obj.y),m_obj.width,m_obj.height,
                               edgecolor=edge_c,facecolor=face_c,fill=True,lw=1)
            ax1.add_patch(rect)
            ax1.text(m_obj.x+m_obj.width/2,m_obj.y+m_obj.height/2,
                     f'{m_obj.name}{text_suffix}',ha='center',va='center',fontsize=6) 

        if self.bound:
            bound_rect=plt.Rectangle((0,0),self.bound.width,self.bound.height,
                                    edgecolor='red',facecolor='none',lw=2, label='칩 경계') 
            ax1.add_patch(bound_rect)
        ax1.legend(fontsize='small') 

        ax1.set_title(plot_title_p, fontsize=10) 
        ax1.set_xlabel("X 좌표", fontsize=8) 
        ax1.set_ylabel("Y 좌표", fontsize=8) 
        ax1.tick_params(axis='both', which='major', labelsize=7) 
        
        plot_max_w_val = max(self.max_width, self.bound.width if self.bound else 0.0)
        plot_max_h_val = max(self.max_height, self.bound.height if self.bound else 0.0)
        min_x_coord_val = min(m.x for m in self.modules) if self.modules else 0.0
        min_y_coord_val = min(m.y for m in self.modules) if self.modules else 0.0
        
        padding_x_val = (plot_max_w_val - min_x_coord_val) * 0.1 + 50 
        padding_y_val = (plot_max_h_val - min_y_coord_val) * 0.1 + 50 
        ax1.set_xlim(min(0, min_x_coord_val) - padding_x_val, plot_max_w_val + padding_x_val + 150) 
        ax1.set_ylim(min(0, min_y_coord_val) - padding_y_val, plot_max_h_val + padding_y_val + 150) 

        ax1.set_aspect('equal', adjustable='box') 

        plot_title_t = "B*-Tree 구조" 
        if iteration is not None:
             plot_title_t = f"B*-Tree 구조 (Iter={iteration}{title_suffix})"
        
        if self.root : 
            G=nx.DiGraph()
            pos={} 
            self._add_edges_for_nx(self.root,None,0,0,G,pos) 

            nx.draw(G, pos, ax=ax2, with_labels=True, node_color="lightblue",
                    edge_color="black", node_size=1500, font_size=8,  
                    arrows=True, arrowstyle='->', arrowsize=10) 
            ax2.set_title(plot_title_t, fontsize=10) 
        else:
            ax2.set_title("B*-Tree 구조 (표시할 트리 없음)", fontsize=10) 

        plt.tight_layout() 

    def _plot_node(self,btnode,ax): 
        if not btnode:
            return
        m_obj=btnode.module 
        rect=plt.Rectangle((m_obj.x,m_obj.y),m_obj.width,m_obj.height,
                           edgecolor='blue',facecolor='lightblue',fill=True,lw=1) 
        ax.add_patch(rect)
        ax.text(m_obj.x+m_obj.width/2,m_obj.y+m_obj.height/2,
                f'{m_obj.name}\n#{m_obj.order if m_obj.order is not None else "N/A"}',ha='center',va='center',fontsize=7) 
        self._plot_node(btnode.left,ax) 
        self._plot_node(btnode.right,ax) 

    def _add_edges_for_nx(self,current_btnode, parent_module_name, depth, current_x_pos, G, pos_dict):
        if not current_btnode:
            return current_x_pos 

        current_module_name = current_btnode.module.name
        
        effective_x_after_left = current_x_pos
        if current_btnode.left:
            effective_x_after_left = self._add_edges_for_nx(current_btnode.left, current_module_name, depth + 1, current_x_pos, G, pos_dict)
        
        current_node_x = effective_x_after_left 
        pos_dict[current_module_name] = (current_node_x, -depth * 2) 
        G.add_node(current_module_name,label=current_module_name) 
        if parent_module_name:
            G.add_edge(parent_module_name, current_module_name)

        effective_x_after_right = current_node_x + 1 
        if current_btnode.right:
            effective_x_after_right = self._add_edges_for_nx(current_btnode.right, current_module_name, depth + 1, effective_x_after_right, G, pos_dict)
            return effective_x_after_right 
        
        return effective_x_after_right 

# ─────────────────────────────────────────────────────────────
# HPWL, 면적, penalty 함수들 
# ─────────────────────────────────────────────────────────────

def calculate_hpwl(modules): 
    net_dict={} 
    for m_obj in modules:
        for net_name_str in m_obj.net:
            if net_name_str not in net_dict:
                net_dict[net_name_str]=[]
            net_dict[net_name_str].append(m_obj) 
    if not net_dict: 
        return 0.0
    total_hpwl_val=0.0
    for net_name_key, mods_in_net_list in net_dict.items(): 
        if not mods_in_net_list: continue 
        min_x_val=min(m_o.x for m_o in mods_in_net_list)
        max_x_val=max(m_o.x+m_o.width for m_o in mods_in_net_list)
        min_y_val=min(m_o.y for m_o in mods_in_net_list)
        max_y_val=max(m_o.y+m_o.height for m_o in mods_in_net_list)
        net_hpwl_val=(max_x_val - min_x_val)+(max_y_val - min_y_val)
        total_hpwl_val+=net_hpwl_val
    return total_hpwl_val

def calculate_total_area(modules): 
    if not modules:
        return (0.0,0.0,0.0) 
    
    min_x_coord=min(m.x for m in modules) if modules else 0.0 
    max_x_coord=max(m.x+m.width for m in modules) if modules else 0.0
    min_y_coord=min(m.y for m in modules) if modules else 0.0
    max_y_coord=max(m.y+m.height for m in modules) if modules else 0.0
    width_val=(max_x_coord - min_x_coord)
    height_val=(max_y_coord - min_y_coord)
    return (width_val,height_val,(width_val*height_val))

def calc_combined_cost(modules, w=Global_w, chip=None, r_penalty=Global_r_penalty, r_dead_space=Global_r_dead_space, use_dead_space_in_cost=False, return_all=False):
    cost_scale_factor=0.001 
    if not modules: 
        if return_all:
            return (0.0, 0.0, 0.0, 0.0, 0.0) 
        else:
            return 0.0

    base_area_scale=sum(m.area for m in modules) 
    if base_area_scale<1e-12: 
        base_area_scale=1.0
    
    net_connected_area_sum=sum(m.area for m in modules if m.net)
    if net_connected_area_sum<1e-12: 
        net_connected_area_sum=base_area_scale 

    bbox_w,bbox_h,bbox_area=calculate_total_area(modules) 
    hpwl_val=calculate_hpwl(modules)
    
    area_normalized   = bbox_area/base_area_scale if base_area_scale > 1e-9 else bbox_area
    hpwl_normalized   = hpwl_val/(2*math.sqrt(net_connected_area_sum)) if net_connected_area_sum > 1e-9 else hpwl_val 
    
    area_normalized   *=500 

    # Penalty 계산 (항상 계산은 하되, 비용 함수에 포함 여부는 use_dead_space_in_cost와 별개로 r_penalty로 조절)
    penalty_total_sum_val=0.0 
    if chip and chip.bound: 
        chip_boundary_w = chip.bound.width
        chip_boundary_h = chip.bound.height
        
        area_violation_val = 0.0
        if bbox_w > chip_boundary_w and bbox_h > chip_boundary_h: 
            area_violation_val = (bbox_w * bbox_h) - (chip_boundary_w * chip_boundary_h)
        elif bbox_w > chip_boundary_w: 
            area_violation_val = (bbox_w - chip_boundary_w) * bbox_h 
        elif bbox_h > chip_boundary_h: 
            area_violation_val = (bbox_h - chip_boundary_h) * bbox_w 

        length_violation_val=0.0
        for m_obj_penalty in modules:
            if m_obj_penalty.x + m_obj_penalty.width > chip_boundary_w:
                length_violation_val += ( (m_obj_penalty.x + m_obj_penalty.width) - chip_boundary_w )**2
            if m_obj_penalty.y + m_obj_penalty.height > chip_boundary_h:
                length_violation_val += ( (m_obj_penalty.y + m_obj_penalty.height) - chip_boundary_h )**2
            if m_obj_penalty.x < 0: 
                length_violation_val += (m_obj_penalty.x)**2 
            if m_obj_penalty.y < 0:
                length_violation_val += (m_obj_penalty.y)**2
        penalty_total_sum_val=area_violation_val+length_violation_val
    
    penalty_normalized_val = penalty_total_sum_val/base_area_scale if base_area_scale > 1e-9 else penalty_total_sum_val 
    
    # Dead Space 계산 (항상 계산)
    total_module_actual_area = sum(m.area for m in modules)
    dead_space_absolute_val = bbox_area - total_module_actual_area
    if bbox_area > 1e-9:
        dead_space_normalized_val = (dead_space_absolute_val / bbox_area) * 100.0
    else:
        dead_space_normalized_val = 0.0

    # 최종 비용 계산
    cost_final_val = w * area_normalized + (1 - w) * hpwl_normalized + r_penalty * penalty_normalized_val
    
    if use_dead_space_in_cost: # Dead Space를 비용에 추가하는 경우
        cost_final_val += r_dead_space * dead_space_normalized_val
    
    cost_final_val *= cost_scale_factor 

    if return_all:
        return (cost_final_val, area_normalized, hpwl_normalized, penalty_normalized_val, dead_space_normalized_val)
    else:
        return cost_final_val

# ─────────────────────────────────────────────────────────────
# Q-Learning / fast_sa 
# ─────────────────────────────────────────────────────────────

Q_TABLE={} 
ACTIONS=["rotate","move","swap"] 
epsilon_q=0.3 
alpha_q=0.5   
gamma_q=0.9   

def get_state_key(chip_obj, r_penalty_state=Global_r_penalty, r_dead_space_state=Global_r_dead_space, use_ds_in_cost_state=False): 
    c_val_state=calc_combined_cost(chip_obj.modules, w=Global_w, chip=chip_obj, 
                                   r_penalty=r_penalty_state, r_dead_space=r_dead_space_state, 
                                   use_dead_space_in_cost=use_ds_in_cost_state, 
                                   return_all=False)
    return round(c_val_state,1) 

def select_action_by_q(state_key_str):
    if random.random()<epsilon_q: 
        return random.randrange(len(ACTIONS))
    else: 
        if state_key_str not in Q_TABLE: 
            Q_TABLE[state_key_str]=[0.0,0.0,0.0]
        q_values_arr=Q_TABLE[state_key_str] 
        return q_values_arr.index(max(q_values_arr)) 

def update_q_value(s_key_str,a_idx_val,reward_val,s_next_key_str):
    if s_key_str not in Q_TABLE:
        Q_TABLE[s_key_str]=[0.0,0.0,0.0]
    if s_next_key_str not in Q_TABLE: 
        Q_TABLE[s_next_key_str]=[0.0,0.0,0.0]
    
    q_old_val=Q_TABLE[s_key_str][a_idx_val] 
    max_q_next_val=max(Q_TABLE[s_next_key_str]) 
    
    q_new_val=q_old_val+alpha_q*(reward_val+gamma_q*max_q_next_val-q_old_val) 
    Q_TABLE[s_key_str][a_idx_val]=q_new_val


def fast_sa(chip_obj,max_iter=5000,P_initial=0.99,c_cooling=20, 
            w_cost_sa=Global_w, sample_moves_num=10, 
            r_penalty_sa=Global_r_penalty, r_dead_space_sa=Global_r_dead_space, 
            use_ds_in_cost_sa=False): 
    
    original_state_chip=copy.deepcopy(chip_obj) 
    original_cost_val=calc_combined_cost(chip_obj.modules,w_cost_sa,chip=chip_obj,
                                         r_penalty=r_penalty_sa, r_dead_space=r_dead_space_sa, 
                                         use_dead_space_in_cost=use_ds_in_cost_sa,
                                         return_all=False)
    uphill_differences=[] 
    
    temp_chip_for_t1 = copy.deepcopy(chip_obj) 
    for _ in range(sample_moves_num+10): 
        current_state_key_t1=get_state_key(temp_chip_for_t1, r_penalty_sa, r_dead_space_sa, use_ds_in_cost_sa) 
        action_index_t1=select_action_by_q(current_state_key_t1) 
        action_name_t1=ACTIONS[action_index_t1]

        old_cost_t1=calc_combined_cost(temp_chip_for_t1.modules,w_cost_sa,chip=temp_chip_for_t1,
                                       r_penalty=r_penalty_sa, r_dead_space=r_dead_space_sa, 
                                       use_dead_space_in_cost=use_ds_in_cost_sa,
                                       return_all=False)
        msg_t1,op_t1,op_data_t1=temp_chip_for_t1.apply_specific_operation(action_name_t1) 
        new_cost_t1=calc_combined_cost(temp_chip_for_t1.modules,w_cost_sa,chip=temp_chip_for_t1,
                                       r_penalty=r_penalty_sa, r_dead_space=r_dead_space_sa, 
                                       use_dead_space_in_cost=use_ds_in_cost_sa,
                                       return_all=False)
        delta_e_t1=new_cost_t1-old_cost_t1 

        reward_t1=(old_cost_t1-new_cost_t1) 
        next_state_key_t1=get_state_key(temp_chip_for_t1, r_penalty_sa, r_dead_space_sa, use_ds_in_cost_sa) 
        update_q_value(current_state_key_t1,action_index_t1,reward_t1,next_state_key_t1) 

        if delta_e_t1>0: 
            uphill_differences.append(delta_e_t1)
        temp_chip_for_t1=copy.deepcopy(original_state_chip) 

    avg_uphill_delta=1.0 
    if uphill_differences:
        avg_uphill_delta=sum(uphill_differences)/len(uphill_differences) 
    if avg_uphill_delta<1e-12: 
        avg_uphill_delta=1.0

    temp_t1_scale_factor=1.0 
    temp_t1_initial=abs(avg_uphill_delta/math.log(P_initial))*temp_t1_scale_factor 
    cost_type_msg = "DeadSpace 및 Penalty 포함" if use_ds_in_cost_sa else "Penalty만 포함"
    print(f"초기 온도 T1={temp_t1_initial:.3f} (비용함수: {cost_type_msg})")


    current_best_chip=copy.deepcopy(chip_obj) 
    current_best_cost=original_cost_val       
    current_iteration_cost =original_cost_val 
    current_chip_state = chip_obj             

    temperatures_log=[] 
    for n_iter in range(1,max_iter+1): 
        chip_for_this_iteration = copy.deepcopy(current_chip_state)
        cost_before_moves = current_iteration_cost 

        cost_differences_local=[] 
        best_local_move_cost=float('inf') 
        best_local_operation_data=None 
        
        temp_chip_for_local_search = copy.deepcopy(chip_for_this_iteration) 
        for _ in range(sample_moves_num): 
            state_key_local=get_state_key(temp_chip_for_local_search, r_penalty_sa, r_dead_space_sa, use_ds_in_cost_sa) 
            action_index_local=select_action_by_q(state_key_local) 
            action_name_local=ACTIONS[action_index_local] 

            cost_before_local_move=calc_combined_cost(temp_chip_for_local_search.modules,w_cost_sa,chip=temp_chip_for_local_search,
                                                      r_penalty=r_penalty_sa, r_dead_space=r_dead_space_sa, 
                                                      use_dead_space_in_cost=use_ds_in_cost_sa, return_all=False)
            msg_local,op_local,op_data_local=temp_chip_for_local_search.apply_specific_operation(action_name_local) 
            cost_after_local_move=calc_combined_cost(temp_chip_for_local_search.modules,w_cost_sa,chip=temp_chip_for_local_search,
                                                     r_penalty=r_penalty_sa, r_dead_space=r_dead_space_sa, 
                                                     use_dead_space_in_cost=use_ds_in_cost_sa, return_all=False)

            reward_local=(cost_before_local_move-cost_after_local_move) 
            next_state_key_local=get_state_key(temp_chip_for_local_search, r_penalty_sa, r_dead_space_sa, use_ds_in_cost_sa) 
            update_q_value(state_key_local,action_index_local,reward_local,next_state_key_local) 

            delta_e_from_iter_start_local=cost_after_local_move - cost_before_moves 
            cost_differences_local.append(abs(delta_e_from_iter_start_local)) 

            if cost_after_local_move < best_local_move_cost: 
                best_local_move_cost=cost_after_local_move
                best_local_operation_data=op_data_local 
            
            temp_chip_for_local_search=copy.deepcopy(chip_for_this_iteration) 
        
        avg_cost_difference_local=1e-6 
        if cost_differences_local: 
            avg_cost_difference_local=max(sum(cost_differences_local)/len(cost_differences_local),1e-6) 
        
        current_temp_t = 1e-6
        if n_iter==1: 
            current_temp_t=temp_t1_initial
        elif 2<=n_iter<=300: 
            current_temp_t=max((temp_t1_initial*avg_cost_difference_local)/(n_iter*c_cooling),1e-6) 
        else: 
            current_temp_t=max((temp_t1_initial*avg_cost_difference_local)/n_iter,1e-6) 
        temperatures_log.append(current_temp_t) 

        chip_before_main_move = copy.deepcopy(current_chip_state) 
        cost_before_main_move = current_iteration_cost          
        
        if best_local_operation_data:
            reapply_msg, reapply_op_type = current_chip_state.reapply_operation(best_local_operation_data)
        else: 
            state_key_random_move = get_state_key(current_chip_state, r_penalty_sa, r_dead_space_sa, use_ds_in_cost_sa)
            action_index_random_move = select_action_by_q(state_key_random_move)
            action_name_random_move = ACTIONS[action_index_random_move]
            reapply_msg, reapply_op_type, best_local_operation_data = current_chip_state.apply_specific_operation(action_name_random_move)


        cost_after_main_move=calc_combined_cost(current_chip_state.modules,w_cost_sa,chip=current_chip_state,
                                                r_penalty=r_penalty_sa, r_dead_space=r_dead_space_sa, 
                                                use_dead_space_in_cost=use_ds_in_cost_sa, return_all=False)
        delta_e_main_move=cost_after_main_move - cost_before_main_move 
        
        acceptance_probability = 0.0 
        acceptance_status_str = "REJECT" 
        if delta_e_main_move < 0: 
            current_iteration_cost = cost_after_main_move
            if cost_after_main_move < current_best_cost: 
                current_best_cost = cost_after_main_move
                current_best_chip = copy.deepcopy(current_chip_state)
            acceptance_status_str="ACCEPT (개선)"
            acceptance_probability=1.0
        else: 
            if current_temp_t < 1e-12: 
                acceptance_probability=0.0
            else:
                acceptance_probability=math.exp(-abs(delta_e_main_move)/current_temp_t) 
            
            if random.random() < acceptance_probability: 
                current_iteration_cost = cost_after_main_move
                acceptance_status_str="ACCEPT (악화)" 
            else: 
                current_chip_state = copy.deepcopy(chip_before_main_move) 
                current_iteration_cost = cost_before_main_move 
                acceptance_status_str="REJECT" 
        
        if n_iter % 100 == 0 or n_iter == 1 or n_iter == max_iter : 
            op_name_disp = best_local_operation_data.get('op') if best_local_operation_data else 'N/A'
            print(f"[Iter={n_iter:4d}] T={current_temp_t:8.4f} | Cost={current_iteration_cost:8.3f} (Best={current_best_cost:8.3f}) | dE={delta_e_main_move:8.3f} | Prob={acceptance_probability:6.4f} | {acceptance_status_str} | Op: {op_name_disp}")
            
    plt.figure(figsize=(10,6)) 
    plt.plot(range(1,max_iter+1),temperatures_log)
    plt.xlabel("반복 횟수") 
    plt.ylabel("온도")   
    plt.title("SA 온도 변화 추이") 
    plt.grid(True)
    #plt.show()

    return current_best_chip 

def partial_sa_for_initial(chip_obj, pre_iter=500): 
    print("\n[Info] --- 초기 레이아웃 개선을 위한 부분 SA 실행 ---") 
    improved_chip_obj = fast_sa(
        chip_obj,
        max_iter=pre_iter, 
        P_initial=0.95, 
        c_cooling=50,  
        w_cost_sa=Global_w, 
        sample_moves_num=15, 
        r_penalty_sa=Global_r_penalty, # 부분 SA는 기본 페널티 가중치 사용      
        use_ds_in_cost_sa=False # 부분 SA는 Dead Space 비용 사용 안함
    )
    print("[Info] --- 부분 SA 종료. 개선된 초기 레이아웃 사용 ---") 
    return improved_chip_obj

# ─────────────────────────────────────────────────────────────
# 4. 파일 파싱(Yal, GSRC) 
# ─────────────────────────────────────────────────────────────

def parse_yal(file_path_str):
    modules_list=[] 
    with open(file_path_str,'r', encoding='utf-8') as f_yal: 
        lines_yal=f_yal.readlines()
    current_module_data_dict={}
    is_in_network_section=False 
    for line_str in lines_yal:
        line_str=line_str.strip()
        if not line_str or line_str.startswith("/*") or line_str.startswith("*"): 
            continue
        
        if line_str.startswith("MODULE"):
            parts_module = line_str.split()
            if len(parts_module) > 1:
                module_name_str=parts_module[1].strip(';')
                current_module_data_dict={'name':module_name_str,'net':[]} 
            continue
        
        if line_str.startswith("TYPE"):
            parts_type = line_str.split()
            if len(parts_type) > 1:
                module_type_str=parts_type[1].strip(';')
                current_module_data_dict['type']=module_type_str
            continue

        if line_str.startswith("DIMENSIONS"):
            dim_str_val = line_str.replace("DIMENSIONS","").replace(";","").strip()
            coords_val = []
            if '(' in dim_str_val and ')' in dim_str_val:
                points_str_list = re.findall(r'\([\d\.\s,]+\)', dim_str_val)
                for p_str_item in points_str_list:
                    try:
                        x_coord_dim, y_coord_dim = map(float, p_str_item.strip('()').split(','))
                        coords_val.append(x_coord_dim)
                        coords_val.append(y_coord_dim)
                    except ValueError:
                        pass
            
            dims_list = coords_val if coords_val else list(map(float,dim_str_val.split()))

            if len(dims_list) == 2: 
                current_module_data_dict['width']=dims_list[0]
                current_module_data_dict['height']=dims_list[1]
            elif len(dims_list) >= 4: 
                x_coords_list=[dims_list[i] for i in range(0,len(dims_list),2)]
                y_coords_list=[dims_list[i+1] for i in range(0,len(dims_list),2)]
                if x_coords_list and y_coords_list:
                    current_module_data_dict['width']=max(x_coords_list)-min(x_coords_list)
                    current_module_data_dict['height']=max(y_coords_list)-min(y_coords_list)
                else: 
                    current_module_data_dict['width']=0.0
                    current_module_data_dict['height']=0.0
            else: 
                current_module_data_dict['width']=0.0
                current_module_data_dict['height']=0.0
            continue
            
        if line_str.startswith("NETWORK"):
            is_in_network_section=True
            continue
        if line_str.startswith("ENDNETWORK"):
            is_in_network_section=False
            continue
            
        if line_str.startswith("ENDMODULE"):
            if all(k in current_module_data_dict for k in ['name', 'width', 'height', 'type']):
                 modules_list.append(Module(
                    name=current_module_data_dict['name'],
                    width=current_module_data_dict['width'],
                    height=current_module_data_dict['height'],
                    module_type=current_module_data_dict['type'],
                    net=current_module_data_dict.get('net',[]) 
                ))
            current_module_data_dict={} 
            continue
            
        if is_in_network_section:
            current_module_data_dict['net'].append(line_str.strip(';'))
            continue
            
    return modules_list

def parse_gsrc_blocks(blocks_file_path):
    modules_list_gsrc=[] 
    gsrc_block_pattern=re.compile(r"^(sb\d+)\s+\S+\s+(\d+)\s+(.*)$") 
    with open(blocks_file_path,'r', encoding='utf-8') as f_gsrc_blocks: 
        lines_gsrc_blocks=f_gsrc_blocks.readlines()
    for line_gsrc_block in lines_gsrc_blocks:
        line_gsrc_block=line_gsrc_block.strip()
        
        if not line_gsrc_block or line_gsrc_block.startswith('#') or line_gsrc_block.startswith('UCSC'):
            continue
        if line_gsrc_block.startswith('p') and 'terminal' in line_gsrc_block: 
            continue

        match_gsrc_block=gsrc_block_pattern.match(line_gsrc_block)
        if match_gsrc_block:
            block_name_str=match_gsrc_block.group(1)
            coords_str_gsrc=match_gsrc_block.group(3)
            
            gsrc_coord_pattern=re.compile(r"\(([\-\d\.]+)\s*,\s*([\-\d\.]+)\)") 
            found_coords_gsrc=gsrc_coord_pattern.findall(coords_str_gsrc)
            
            x_coords_gsrc,y_coords_gsrc=[],[]
            for (sx_gsrc,sy_gsrc) in found_coords_gsrc:
                try:
                    x_coords_gsrc.append(float(sx_gsrc))
                    y_coords_gsrc.append(float(sy_gsrc))
                except ValueError:
                    print(f"[Warning] {block_name_str}의 좌표 파싱 불가: ({sx_gsrc},{sy_gsrc})") 
                    continue 
            
            if not x_coords_gsrc or not y_coords_gsrc: 
                print(f"[Warning] {block_name_str}에 대한 유효한 좌표 없음") 
                continue

            min_x_gsrc,max_x_gsrc=min(x_coords_gsrc),max(x_coords_gsrc)
            min_y_gsrc,max_y_gsrc=min(y_coords_gsrc),max(y_coords_gsrc)
            width_gsrc=max_x_gsrc-min_x_gsrc
            height_gsrc=max_y_gsrc-min_y_gsrc
            modules_list_gsrc.append(Module(block_name_str,width_gsrc,height_gsrc,'BLOCK',net=[])) 
    return modules_list_gsrc

def parse_gsrc_nets(nets_file_path, modules_list_input): 
    module_name_to_obj_map={m.name:m for m in modules_list_input} 
    line_index=0 
    with open(nets_file_path,'r', encoding='utf-8') as f_gsrc_nets: 
        lines_gsrc_nets=f_gsrc_nets.readlines()
    
    current_net_id_counter=0 
    while line_index<len(lines_gsrc_nets):
        line_gsrc_net=lines_gsrc_nets[line_index].strip()
        
        if not line_gsrc_net or line_gsrc_net.startswith('#') or line_gsrc_net.startswith('UCLA'):
            line_index+=1
            continue
        
        if line_gsrc_net.startswith('NetDegree'):
            parts_net_degree=line_gsrc_net.split(':')
            degree_str_val=(parts_net_degree[1].strip() if len(parts_net_degree)>1 else "0")
            try:
                degree_val=int(degree_str_val)
            except ValueError: 
                print(f"[Warning] NetDegree 파싱 불가: {line_gsrc_net}") 
                degree_val=0 
            
            current_net_name_str=f"Net{current_net_id_counter}" 
            current_net_id_counter+=1
            
            pins_in_this_net=[] 
            for _ in range(degree_val): 
                line_index+=1
                if line_index>=len(lines_gsrc_nets): break 
                pin_line_str=lines_gsrc_nets[line_index].strip()
                pin_parts_list=pin_line_str.split() 
                if pin_parts_list:
                    pin_name_str=pin_parts_list[0]
                    if pin_name_str in module_name_to_obj_map: 
                        pins_in_this_net.append(pin_name_str)
                    
            for pin_name_item_str in pins_in_this_net:
                module_name_to_obj_map[pin_name_item_str].net.append(current_net_name_str)
        line_index+=1
    return modules_list_input 

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
        print("[Error] 로드된 모듈 없음. 종료합니다.") 
        sys.exit(1)

    chip_main=Chip(modules) 
    chip_main.calculate_coordinates() 
    
    print("초기 랜덤 B*-Tree 레이아웃 플로팅 중...") 
    chip_main.plot_b_tree(iteration="Initial_Random", title_suffix=" (Random)")
    plt.show()

    init_cost_val, init_aN,init_hN,init_pN, init_dsN = calc_combined_cost(
        chip_main.modules, w=Global_w, chip=chip_main, 
        r_penalty=Global_r_penalty, r_dead_space=Global_r_dead_space, 
        use_dead_space_in_cost=False, return_all=True
    )
    init_w_val,init_h_val,init_area_val=calculate_total_area(chip_main.modules)
    init_hpwl_val =calculate_hpwl(chip_main.modules)
    print("=== 초기 랜덤 Chip 상태 ===") 
    print(f"경계 상자: W={init_w_val:.2f}, H={init_h_val:.2f}, 면적={init_area_val:.2f}")
    print(f"HPWL (절대값)             = {init_hpwl_val:.2f}")
    print(f"정규화된 면적             = {init_aN:.3f}")
    print(f"정규화된 HPWL             = {init_hN:.3f}")
    print(f"정규화된 페널티          = {init_pN:.3f}")
    print(f"정규화된 DeadSpace       = {init_dsN:.3f}") 
    print(f"초기 비용 (w={Global_w:.2f}, r_penalty={Global_r_penalty:.2f}, 페널티만 사용) = {init_cost_val:.3f}")

    run_partial_sa_input = input("초기 레이아웃 개선을 위해 부분 SA를 실행하시겠습니까? (y/n): ")
    if run_partial_sa_input.lower().startswith('y'):
        chip_main = partial_sa_for_initial(chip_main, pre_iter=500) 
        print("부분 SA 후 레이아웃 플로팅 중...") 
        chip_main.plot_b_tree(iteration="After_Partial_SA", title_suffix=" (Partial SA)")
        #plt.show() 

        partial_sa_cost_val, p_aN,p_hN,p_pN, p_dsN = calc_combined_cost(
            chip_main.modules, w=Global_w, chip=chip_main, 
            r_penalty=Global_r_penalty, r_dead_space=Global_r_dead_space, 
            use_dead_space_in_cost=False, return_all=True
        )
        p_w,p_h,p_area=calculate_total_area(chip_main.modules)
        p_hpwl=calculate_hpwl(chip_main.modules)
        print("=== 부분 SA 후 Chip 상태 ===") 
        print(f"경계 상자: W={p_w:.2f}, H={p_h:.2f}, 면적={p_area:.2f}")
        print(f"HPWL (절대값)             = {p_hpwl:.2f}")
        print(f"정규화된 면적             = {p_aN:.3f}")
        print(f"정규화된 HPWL             = {p_hN:.3f}")
        print(f"정규화된 페널티          = {p_pN:.3f}")
        print(f"정규화된 DeadSpace       = {p_dsN:.3f}")
        print(f"부분 SA 후 비용 (페널티만 사용) = {partial_sa_cost_val:.3f}")

    # [C] 1단계 메인 SA 실행 (페널티만 사용)
    print("\n[Info] 1단계 전체 FastSA (Q-Learning, 페널티 비용) 최적화 진행 중...") 
    first_sa_best_chip = fast_sa(chip_main, 
                                max_iter=2000, 
                                P_initial=0.95,            
                                c_cooling=100,          
                                w_cost_sa=Global_w,        
                                sample_moves_num=30,  
                                r_penalty_sa=Global_r_penalty, 
                                r_dead_space_sa=0, # 1단계에서는 Dead Space 가중치 0
                                use_ds_in_cost_sa=False) # 1단계에서는 Dead Space 비용 사용 안함          

    print("1단계 SA 최적화 레이아웃 플로팅 중 (Compaction 전)...") 
    first_sa_best_chip.plot_b_tree(iteration="1st_SA_Result", title_suffix=" (Before Compact)")
    #plt.show()

    print("\n[Info] 1단계 SA 결과에 Compaction 수행 중...") 
    # compact_floorplan_final은 first_sa_best_chip 객체를 직접 수정합니다.
    first_sa_best_chip.compact_floorplan_final()

    print("1단계 SA 및 Compaction 후 레이아웃 플로팅 중...") 
    first_sa_best_chip.plot_b_tree(iteration="1st_SA_Compacted", title_suffix=" (After Compact)")
    #plt.show()
    
    # 1단계 SA + Compaction 후 비용 (Dead Space 포함하여 참고용으로 출력)
    cost_after_1st_sa_compact, an_1, hn_1, pn_1, dsn_1 = calc_combined_cost(
        first_sa_best_chip.modules, w=Global_w, chip=first_sa_best_chip,
        r_penalty=Global_r_penalty, r_dead_space=Global_r_dead_space,
        use_dead_space_in_cost=True, return_all=True # 모든 항을 포함하여 비용 계산
    )
    w_1, h_1, area_1 = calculate_total_area(first_sa_best_chip.modules)
    hpwl_1 = calculate_hpwl(first_sa_best_chip.modules)
    total_mod_area_1 = sum(m.area for m in first_sa_best_chip.modules)
    ds_abs_1 = area_1 - total_mod_area_1
    ds_perc_1 = (ds_abs_1 / area_1) * 100 if area_1 > 1e-9 else 0.0

    print("\n=== 1단계 SA + Compaction 후 상태 (참고용 Dead Space 포함 비용) ===")
    print(f"경계 상자: W={w_1:.2f}, H={h_1:.2f}, 면적={area_1:.2f}")
    print(f"HPWL (절대값) = {hpwl_1:.2f}, 정규화된 HPWL = {hn_1:.3f}")
    print(f"정규화된 면적 = {an_1:.3f}, 정규화된 페널티 = {pn_1:.3f}")
    print(f"정규화된 DeadSpace = {dsn_1:.3f}, 실제 DeadSpace = {ds_abs_1:.2f} ({ds_perc_1:.2f}%)")
    print(f"비용 (모든 항 포함) = {cost_after_1st_sa_compact:.3f}")


    # [D] 2단계 메인 SA 실행 (Compaction된 결과로부터 시작, Dead Space 및 Penalty 모두 사용)
    print("\n[Info] Compaction된 결과를 사용하여 2단계 전체 FastSA (Dead Space 및 Penalty 비용) 진행 중...") 
    # 2단계 SA는 first_sa_best_chip (이미 컴팩션됨)을 입력으로 사용
    second_sa_best_chip = fast_sa(first_sa_best_chip, 
                                 max_iter=2000,    
                                 P_initial=0.95,            
                                 c_cooling=100, 
                                 w_cost_sa=Global_w,         
                                 sample_moves_num=30,   
                                 r_penalty_sa=Global_r_penalty, # 2단계에서도 페널티 가중치 적용
                                 r_dead_space_sa=Global_r_dead_space, # Dead Space 가중치 적용
                                 use_ds_in_cost_sa=True) # Dead Space 비용 사용 활성화         

    print("2단계 SA 최적화 레이아웃 플로팅 중 (최종 Compaction 전)...") 
    second_sa_best_chip.plot_b_tree(iteration="2nd_SA_Result", title_suffix=" (Before Final Compact)")
    #plt.show()
    
    print("\n[Info] 2단계 SA 결과에 최종 Compaction 수행 중...") 
    second_sa_best_chip.compact_floorplan_final()

    print("\n=== 최종 Compaction 후 (2단계 SA 결과 기반) ===") 
    final_cost, final_aN, final_hN, final_pN, final_dsN = calc_combined_cost(
        second_sa_best_chip.modules, w=Global_w, chip=second_sa_best_chip, 
        r_penalty=Global_r_penalty, r_dead_space=Global_r_dead_space, 
        use_dead_space_in_cost=True, return_all=True 
    )
    final_w, final_h, final_area = calculate_total_area(second_sa_best_chip.modules)
    final_hpwl = calculate_hpwl(second_sa_best_chip.modules)
    final_total_mod_area = sum(m.area for m in second_sa_best_chip.modules)
    actual_dead_space_area = final_area - final_total_mod_area
    actual_dead_space_percent = (actual_dead_space_area / final_area) * 100 if final_area > 1e-9 else 0.0

    print(f"최종 Compaction 후 경계 상자: W={final_w:.2f}, H={final_h:.2f}, 면적={final_area:.2f}")
    print(f"최종 Compaction 후 HPWL (절대값)         = {final_hpwl:.2f}")
    print(f"최종 Compaction 후 정규화된 면적         = {final_aN:.3f}") 
    print(f"최종 Compaction 후 정규화된 HPWL         = {final_hN:.3f}")
    print(f"최종 Compaction 후 정규화된 페널티      = {final_pN:.3f}") 
    print(f"최종 Compaction 후 정규화된 DeadSpace  = {final_dsN:.3f}")
    print(f"최종 Compaction 후 실제 DeadSpace 면적 = {actual_dead_space_area:.2f} ({actual_dead_space_percent:.2f}%)")
    print(f"최종 Compaction 후 비용 (w_area={Global_w:.2f}, r_penalty={Global_r_penalty:.2f}, r_ds={Global_r_dead_space:.2f}) = {final_cost:.3f}")


    print("최종 Compaction 후 레이아웃 플로팅 중...") 
    second_sa_best_chip.plot_b_tree(iteration="Final_Compacted_Layout", title_suffix=" (Final Compacted)") 
    plt.show()

    print("\n플로어플래닝 프로세스 완료.") 
