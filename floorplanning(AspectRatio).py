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
#7/09 K-Parent Based Search를 적용한 Simulated Annealing 플로어플래닝 (멀티프로세싱 버전)
#     Q-Learning 제거 버전 - 완전 랜덤 액션 선택 (33.33%씩)
#     기존 fast_sa 함수를 multiprocess_k_parent_sa로 대체
#     기존 코드 구조와 함수명 최대한 유지
#7/11 요청에 따라 백업용 단일 프로세스 함수(k_parent_sa_single_process) 제거
#7/12 요청에 따라 온도 설정 및 냉각 스케줄링을 참조 코드를 기반으로 수정. 병렬 처리 로직 수정.
#7/15 [개선안 적용] 하이브리드 탐색(병렬+지역), 1단계 SA 강화, 중간 Compaction 제거
#7/15 [개선안 적용] 3단계 SA(깊은 단일 탐색) 추가
#7/15 [개선안 적용] 주기적 Pruning(가지치기) 및 돌연변이(Mutation) 기능 추가
#7/15 [개선안 적용] 적응형 냉각 스케줄 및 재가열(Reheating) 기능 추가
#7/16 [개선안 적용] 비용 항목 간 스케일 불균형 문제를 해결하기 위한 동적 스케일링(CostScaler) 도입
#7/16 [개선안 적용] 전역 가중치 변수들을 CostWeights 클래스로 통합하여 관리
#7/16 [개선안 적용] 탐색 전략 변경: 매 단계에서 N개의 연산을 시도하고 그 중 가장 좋은 연산을 후보로 선택 (select_best_of_n_moves)
#7/16 [개선안 적용] 워커 내부 프루닝: 워커가 비용 개선을 전혀 못했을 경우, 해당 워커의 모든 시도를 취소하고 초기 상태로 복귀
#7/17 [개선안 적용] 최종 3단계 SA를 병렬 집중 탐색(Exploitation)과 직렬 탐험(Exploration)을 결합한 하이브리드 방식으로 변경
#7/17 [개선안 적용] Parent(입력)와 Bound(계산) 개념을 분리하여 명확성 증대
# [USER REQUEST] 종횡비(Aspect Ratio) 계산식 수정 (R = Height / Width)
# [USER REQUEST] 병렬 최적화 -> 직렬 최적화로 변경. 각 종횡비마다 모든 가용 코어를 사용하여 순차적으로 실행

import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화 (멀티프로세싱 호환성)
import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import random
import re
import sys
import multiprocessing as mp
import pickle
import time
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from matplotlib import font_manager, rc

sys.setrecursionlimit(10000)

# 멀티프로세싱 환경에서 matplotlib 안전하게 사용하기 위한 설정
os.environ['MPLBACKEND'] = 'Agg'
plt.ioff()  # 인터랙티브 모드 비활성화

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

# ─────────────────────────────────────────────────────────────
# 비용 가중치 및 스케일링 클래스 (penalty.py에서 가져옴)
# ─────────────────────────────────────────────────────────────
@dataclass
class CostWeights:
    """비용 함수의 각 항목에 대한 가중치를 관리하는 클래스"""
    w: float = 0.66  # 면적과 HPWL 간의 가중치 (w: 면적, 1-w: HPWL)
    r_penalty: float = 1.0
    r_dead_space: float = 80.0

class CostScaler:
    """각 비용 항목의 스케일을 동적으로 조절하는 클래스"""
    def __init__(self):
        self.area_scale = 1.0
        self.hpwl_scale = 1.0
        self.penalty_scale = 1.0
        self.dead_space_scale = 1.0

    def initialize_scales(self, initial_area_norm, initial_hpwl_norm, initial_penalty_norm, initial_ds_norm):
        """
        초기 정규화된 값들을 기반으로, 각 항목이 비용 계산에서 비슷한 영향력을 갖도록
        스케일링 팩터를 계산합니다.
        """
        target_scale = 100.0
        self.area_scale = target_scale / initial_area_norm if initial_area_norm > 1e-9 else 1.0
        if initial_hpwl_norm > 1e-9:
            self.hpwl_scale = target_scale / initial_hpwl_norm
        else:
            self.hpwl_scale = self.area_scale / 5.0 
        self.penalty_scale = target_scale / initial_penalty_norm if initial_penalty_norm > 1e-9 else 1.0
        self.dead_space_scale = target_scale / initial_ds_norm if initial_ds_norm > 1e-9 else 1.0
        
        # 각 프로세스에서 개별적으로 출력
        print(f"  - 스케일러 초기화: Area={self.area_scale:.2f}, HPWL={self.hpwl_scale:.2f}, Penalty={self.penalty_scale:.2f}, DS={self.dead_space_scale:.2f}")


# K-Parent Based Search를 위한 새로운 클래스들
@dataclass
class ParentState:
    """각 parent state를 관리하는 클래스 (단순화 버전)"""
    chip_state: any  # Chip 객체
    cost: float = float('inf')

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
    def __init__(self, modules, aspect_ratio=1.0):
        self.parent = next((m for m in modules if m.type == 'PARENT'), None)
        self.modules = [m for m in modules if m.type != 'PARENT'] 
        total_area = sum(m.area for m in self.modules)
        
        if total_area > 0:
            base_area = total_area * 1.1
            # [수정] 종횡비 계산 변경 (R = Height / Width)
            # A = W * H = W * (R*W) = R*W^2  => W = sqrt(A/R)
            # H = R * W
            bound_w = math.sqrt(base_area / aspect_ratio)
            bound_h = aspect_ratio * bound_w
        else:
            bound_w = 1000.0
            bound_h = 1000.0

        self.bound = Module(
            name=f'CalculatedBound_R{aspect_ratio:.1f}',
            width=bound_w, 
            height=bound_h, 
            module_type='BOUND'
        )

        self.root = None 
        self.build_b_tree() 
        self.contour_line = [] 
        self.max_width = 0.0
        self.max_height = 0.0


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

    def randomize_b_tree(self, weights, scaler):
        """B*-Tree 구조를 랜덤하게 재구성하고 비용을 반환"""
        self.build_b_tree() 
        self.calculate_coordinates() 
        return calc_combined_cost(self.modules, weights=weights, chip=self, scaler=scaler)

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
        print(f"\n[Info] --- 최종 Greedy Compaction 수행 (R={self.bound.name.split('R')[-1]}) ---")
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
        
        print(f"[Info] --- 최종 Greedy Compaction 완료 (R={self.bound.name.split('R')[-1]}) ---")

    def plot_b_tree(self, iteration=None, title_suffix="", save_only=False): 
        fig_width = 20 
        fig_height = 10 
        if iteration and "Initial_Random" in iteration : 
            fig_width = 24 
            fig_height = 12
        
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(fig_width,fig_height)) 

        plot_title_p = "물리적 배치" 
        if iteration is not None:
            plot_title_p = f"물리적 배치 ({iteration}{title_suffix})"
        
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

            rect=plt.Rectangle((m_obj.x,m_obj.y),m_obj.width,m_obj.height,
                               edgecolor=edge_c,facecolor=face_c,fill=True,lw=1)
            ax1.add_patch(rect)
            ax1.text(m_obj.x+m_obj.width/2,m_obj.y+m_obj.height/2,
                     f'{m_obj.name}{text_suffix}',ha='center',va='center',fontsize=6) 

        # [개선] Parent(입력)와 Bound(계산)를 분리하여 플로팅
        if self.bound:
            bound_rect=plt.Rectangle((0,0),self.bound.width,self.bound.height,
                                    edgecolor='red',facecolor='none',lw=2, label=f'계산된 경계 ({self.bound.name})') 
            ax1.add_patch(bound_rect)
        
        if self.parent:
            parent_rect=plt.Rectangle((0,0),self.parent.width,self.parent.height,
                                    edgecolor='purple',facecolor='none',lw=2, linestyle='--', label='원본 Parent 모듈')
            ax1.add_patch(parent_rect)

        ax1.legend(fontsize='small') 

        ax1.set_title(plot_title_p, fontsize=10) 
        ax1.set_xlabel("X 좌표", fontsize=8) 
        ax1.set_ylabel("Y 좌표", fontsize=8) 
        ax1.tick_params(axis='both', which='major', labelsize=7) 
        
        plot_max_w_val = self.max_width
        plot_max_h_val = self.max_height
        if self.bound:
            plot_max_w_val = max(plot_max_w_val, self.bound.width)
            plot_max_h_val = max(plot_max_h_val, self.bound.height)
        if self.parent:
            plot_max_w_val = max(plot_max_w_val, self.parent.width)
            plot_max_h_val = max(plot_max_h_val, self.parent.height)

        min_x_coord_val = min(m.x for m in self.modules) if self.modules else 0.0
        min_y_coord_val = min(m.y for m in self.modules) if self.modules else 0.0
        
        padding_x_val = (plot_max_w_val - min_x_coord_val) * 0.1 + 50 
        padding_y_val = (plot_max_h_val - min_y_coord_val) * 0.1 + 50 
        ax1.set_xlim(min(0, min_x_coord_val) - padding_x_val, plot_max_w_val + padding_x_val + 150) 
        ax1.set_ylim(min(0, min_y_coord_val) - padding_y_val, plot_max_h_val + padding_y_val + 150) 

        ax1.set_aspect('equal', adjustable='box') 

        plot_title_t = "B*-Tree 구조" 
        if iteration is not None:
             plot_title_t = f"B*-Tree 구조 ({iteration}{title_suffix})"
        
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
        
        # 멀티프로세싱 환경에서 안전하게 저장
        if save_only or iteration:
            filename = f"floorplan_{iteration.replace(' ', '_') if iteration else 'result'}.png"
            try:
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"[Info] 플롯이 {filename}으로 저장되었습니다.")
            except Exception as e:
                print(f"[Warning] 플롯 저장 실패: {e}")
            plt.close(fig)  # 메모리 해제
        else:
            try:
                plt.show()
            except:
                # GUI 백엔드가 없는 경우 자동으로 저장
                filename = "floorplan_result.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"[Info] GUI 표시 실패로 {filename}으로 저장되었습니다.")
                plt.close(fig) 

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

# --- MODIFICATION: CostScaler를 사용하도록 calc_combined_cost 업데이트 ---
def calc_combined_cost(modules, weights: CostWeights, chip=None, scaler: CostScaler = None, use_dead_space_in_cost=False, return_all=False, return_raw_normalized=False):
    if not modules: 
        if return_all:
            return (0.0, 0.0, 0.0, 0.0, 0.0) 
        else:
            return 0.0

    # 1. 기본 값 계산
    base_area_scale=sum(m.area for m in modules) 
    if base_area_scale<1e-12: base_area_scale=1.0
    
    net_connected_area_sum=sum(m.area for m in modules if m.net)
    if net_connected_area_sum<1e-12: net_connected_area_sum=base_area_scale 

    bbox_w,bbox_h,bbox_area=calculate_total_area(modules) 
    hpwl_val=calculate_hpwl(modules)
    
    # 2. 정규화
    area_normalized   = bbox_area/base_area_scale if base_area_scale > 1e-9 else bbox_area
    hpwl_normalized   = hpwl_val/(2*math.sqrt(net_connected_area_sum)) if net_connected_area_sum > 1e-9 else hpwl_val 
    
    # Penalty 계산 (항상 chip.bound 기준)
    penalty_total_sum_val=0.0 
    if chip and chip.bound: 
        chip_boundary_w = chip.bound.width; chip_boundary_h = chip.bound.height
        area_violation_val = max(0, (bbox_w * bbox_h) - (chip_boundary_w * chip_boundary_h))
        length_violation_val=0.0
        for m in modules:
            length_violation_val += max(0, m.x + m.width - chip_boundary_w)**2
            length_violation_val += max(0, m.y + m.height - chip_boundary_h)**2
            length_violation_val += max(0, -m.x)**2
            length_violation_val += max(0, -m.y)**2
        penalty_total_sum_val=area_violation_val+length_violation_val
    
    avg_module_area = base_area_scale / len(modules) if len(modules) > 0 else 1.0
    penalty_normalized_val = penalty_total_sum_val/avg_module_area if avg_module_area > 1e-9 else penalty_total_sum_val 
    
    # Dead Space 계산
    total_module_actual_area = sum(m.area for m in modules)
    dead_space_absolute_val = bbox_area - total_module_actual_area
    dead_space_normalized_val = (dead_space_absolute_val / bbox_area) * 100.0 if bbox_area > 1e-9 else 0.0

    if return_raw_normalized:
        return (area_normalized, hpwl_normalized, penalty_normalized_val, dead_space_normalized_val)

    # 3. 동적 스케일링 적용
    if scaler:
        area_term = area_normalized * scaler.area_scale
        hpwl_term = hpwl_normalized * scaler.hpwl_scale
        penalty_term = penalty_normalized_val * scaler.penalty_scale
        dead_space_term = dead_space_normalized_val * scaler.dead_space_scale
    else: # 스케일러가 없는 경우의 대비책
        area_term = area_normalized * 500
        hpwl_term = hpwl_normalized
        penalty_term = penalty_normalized_val
        dead_space_term = dead_space_normalized_val
    
    # 4. 가중치를 적용하여 최종 비용 계산
    cost_final_val = weights.w * area_term + (1 - weights.w) * hpwl_term + weights.r_penalty * penalty_term
    
    if use_dead_space_in_cost:
        cost_final_val += weights.r_dead_space * dead_space_term
    
    cost_final_val *= 0.01 

    if return_all:
        return (cost_final_val, area_term, hpwl_term, penalty_term, dead_space_term)
    else:
        return cost_final_val

# ─────────────────────────────────────────────────────────────
# 탐색 및 연산 선택 함수
# ─────────────────────────────────────────────────────────────

ACTIONS=["rotate","move","swap"] 

def select_random_action():
    """
    완전 랜덤으로 액션 선택 (각각 33.33% 확률)
    """
    return random.choice(ACTIONS)

def select_best_of_n_moves(current_chip, n_samples, weights, scaler, use_ds_in_cost):
    """
    n_samples 만큼 랜덤 연산을 시도하고, 그 중 가장 비용이 낮은 연산의 결과(chip)와 비용을 반환합니다.
    """
    best_move_chip = None
    best_move_cost = float('inf')

    if n_samples <= 0:
        return current_chip, calc_combined_cost(current_chip.modules, weights=weights, chip=current_chip, scaler=scaler, use_dead_space_in_cost=use_ds_in_cost)

    for _ in range(n_samples):
        temp_chip = copy.deepcopy(current_chip)
        action = select_random_action()
        temp_chip.apply_specific_operation(action)
        
        new_cost = calc_combined_cost(temp_chip.modules, weights=weights, chip=temp_chip, scaler=scaler, use_dead_space_in_cost=use_ds_in_cost)

        if new_cost < best_move_cost:
            best_move_cost = new_cost
            best_move_chip = temp_chip

    if best_move_chip is None:
        return current_chip, calc_combined_cost(current_chip.modules, weights=weights, chip=current_chip, scaler=scaler, use_dead_space_in_cost=use_ds_in_cost)

    return best_move_chip, best_move_cost


# 멀티프로세싱 워커 함수
def worker_sa_depth_search(worker_data):
    """
    각 워커 프로세스에서 실행되는 깊이 탐색 함수.
    """
    try:
        worker_id = worker_data['worker_id']
        parent_chip_state = worker_data['chip_state']
        max_depth = worker_data['max_depth']
        temperature_for_worker = worker_data['temperature']
        weights = worker_data['weights']
        scaler = worker_data['scaler']
        use_ds_in_cost_sa = worker_data['use_ds_in_cost_sa']
        sample_moves_num = worker_data['sample_moves_num']
        
        initial_chip_state = copy.deepcopy(parent_chip_state)
        initial_cost = calc_combined_cost(initial_chip_state.modules, weights=weights, chip=initial_chip_state, scaler=scaler,
                                          use_dead_space_in_cost=use_ds_in_cost_sa)
        
        current_chip = copy.deepcopy(initial_chip_state)
        best_chip_in_worker = copy.deepcopy(initial_chip_state)
        best_cost_in_worker = initial_cost
        has_improved_locally = False

        for _ in range(max_depth):
            cost_before_move = calc_combined_cost(current_chip.modules, weights=weights, chip=current_chip, scaler=scaler,
                                                  use_dead_space_in_cost=use_ds_in_cost_sa)

            chip_after_move, cost_after_move = select_best_of_n_moves(
                current_chip, sample_moves_num, weights, scaler, use_ds_in_cost_sa
            )

            delta_e = cost_after_move - cost_before_move
            accept_move = False
            if delta_e < 0:
                accept_move = True
            elif temperature_for_worker > 1e-12:
                probability = math.exp(-abs(delta_e) / temperature_for_worker)
                accept_move = random.random() < probability

            if accept_move:
                current_chip = chip_after_move
                if cost_after_move < best_cost_in_worker:
                    best_cost_in_worker = cost_after_move
                    best_chip_in_worker = copy.deepcopy(current_chip)
                    if delta_e < 0:
                        has_improved_locally = True

        if not has_improved_locally:
            return {
                'worker_id': worker_id, 'success': True, 'best_chip': initial_chip_state,
                'best_cost': initial_cost, 'error': None
            }
        else:
            return {
                'worker_id': worker_id, 'success': True, 'best_chip': best_chip_in_worker,
                'best_cost': best_cost_in_worker, 'error': None
            }
        
    except Exception as e:
        return {
            'worker_id': worker_data.get('worker_id', -1), 'success': False, 'best_chip': None,
            'best_cost': float('inf'), 'error': str(e)
        }

# 멀티프로세싱 K-Parent Based SA
def multiprocess_k_parent_sa(chip_obj, scaler: CostScaler, weights: CostWeights, max_iter=20000, k_parents=None, max_depth=15,
                            P_initial=0.99, c_cooling=20, sample_moves_num=30,
                            use_ds_in_cost_sa=False, refinement_steps=10,
                            pruning_start_iter=None, pruning_interval=1000, mutation_strength=3,
                            reheating_threshold=None, r_val_for_log=0.0):
    if k_parents is None:
        k_parents = max(1, mp.cpu_count() - 2)
    if k_parents == 0: k_parents = 1
    
    original_state_chip = copy.deepcopy(chip_obj)
    original_cost_val = calc_combined_cost(chip_obj.modules, weights=weights, chip=chip_obj, scaler=scaler,
                                         use_dead_space_in_cost=use_ds_in_cost_sa)
    
    uphill_differences = []
    temp_chip_for_t1 = copy.deepcopy(chip_obj)
    for _ in range(50):
        action_name_t1 = select_random_action()
        old_cost_t1 = calc_combined_cost(temp_chip_for_t1.modules, weights=weights, chip=temp_chip_for_t1, scaler=scaler,
                                       use_dead_space_in_cost=use_ds_in_cost_sa)
        temp_chip_for_t1.apply_specific_operation(action_name_t1)
        new_cost_t1 = calc_combined_cost(temp_chip_for_t1.modules, weights=weights, chip=temp_chip_for_t1, scaler=scaler,
                                       use_dead_space_in_cost=use_ds_in_cost_sa)
        delta_e_t1 = new_cost_t1 - old_cost_t1

        if delta_e_t1 > 0:
            uphill_differences.append(delta_e_t1)
        temp_chip_for_t1 = copy.deepcopy(original_state_chip)

    avg_uphill_delta = 1.0
    if uphill_differences:
        avg_uphill_delta = sum(uphill_differences) / len(uphill_differences)
    if avg_uphill_delta < 1e-12: avg_uphill_delta = 1.0

    temp_t1_initial = abs(avg_uphill_delta / math.log(P_initial)) if P_initial < 1.0 else 1.0

    parent_states = []
    global_best_chip = copy.deepcopy(chip_obj)
    global_best_cost = original_cost_val

    for i in range(k_parents):
        parent_chip = copy.deepcopy(chip_obj)
        for _ in range(random.randint(1, 5)):
            action_name_init = select_random_action()
            parent_chip.apply_specific_operation(action_name_init)
        cost = calc_combined_cost(parent_chip.modules, weights=weights, chip=parent_chip, scaler=scaler,
                                use_dead_space_in_cost=use_ds_in_cost_sa)
        parent_states.append(ParentState(chip_state=copy.deepcopy(parent_chip), cost=cost))

        if cost < global_best_cost:
            global_best_cost = cost
            global_best_chip = copy.deepcopy(parent_chip)

    current_temp_t = temp_t1_initial
    last_improvement_iter = 0

    try:
        with mp.Pool(processes=k_parents) as pool:
            for n_iter in range(1, max_iter + 1):
                best_cost_before_iter = global_best_cost
                
                chip_for_temp_calc = copy.deepcopy(global_best_chip)
                cost_before_moves = global_best_cost
                cost_differences_local = []
                
                temp_chip_for_local_search = copy.deepcopy(chip_for_temp_calc)
                for _ in range(30):
                    action_name_local = select_random_action()
                    temp_chip_for_local_search.apply_specific_operation(action_name_local)
                    cost_after_local_move = calc_combined_cost(temp_chip_for_local_search.modules, weights=weights, chip=temp_chip_for_local_search, scaler=scaler,
                                                             use_dead_space_in_cost=use_ds_in_cost_sa)
                    delta_e_from_iter_start_local = cost_after_local_move - cost_before_moves
                    cost_differences_local.append(abs(delta_e_from_iter_start_local))
                    temp_chip_for_local_search = copy.deepcopy(chip_for_temp_calc)

                avg_cost_difference_local = 1e-6
                if cost_differences_local:
                    avg_cost_difference_local = max(sum(cost_differences_local) / len(cost_differences_local), 1e-6)

                if n_iter == 1:
                    current_temp_t = temp_t1_initial
                else:
                    if reheating_threshold and (n_iter - last_improvement_iter > reheating_threshold):
                        current_temp_t = temp_t1_initial * 0.2
                        last_improvement_iter = n_iter
                    else:
                        if 2 <= n_iter <= 300:
                            current_temp_t = max((temp_t1_initial * avg_cost_difference_local) / (n_iter * c_cooling), 1e-6)
                        else:
                            current_temp_t = max((temp_t1_initial * avg_cost_difference_local) / n_iter, 1e-6)
                
                worker_tasks = []
                for i, parent in enumerate(parent_states):
                    worker_data = {
                        'worker_id': i, 'chip_state': parent.chip_state, 'max_depth': max_depth,
                        'temperature': current_temp_t, 'weights': weights, 'scaler': scaler,
                        'use_ds_in_cost_sa': use_ds_in_cost_sa, 'sample_moves_num': sample_moves_num
                    }
                    worker_tasks.append(worker_data)

                try:
                    worker_results = pool.map(worker_sa_depth_search, worker_tasks)
                    
                    for result in worker_results:
                        if result['success']:
                            worker_id = result['worker_id']
                            parent_states[worker_id] = ParentState(
                                chip_state=copy.deepcopy(result['best_chip']), cost=result['best_cost']
                            )
                            if result['best_cost'] < global_best_cost:
                                global_best_cost = result['best_cost']
                                global_best_chip = copy.deepcopy(result['best_chip'])
                        else:
                            print(f"[Warning][R={r_val_for_log}] Worker {result['worker_id']} 실패: {result['error']}")
                except Exception as e:
                    print(f"[Error][R={r_val_for_log}] Iter {n_iter}: 멀티프로세싱 오류 - {e}")
                    continue

                if global_best_cost < best_cost_before_iter:
                    last_improvement_iter = n_iter

                if n_iter % 200 == 0 or n_iter == 1 or n_iter == max_iter:
                    avg_cost = sum(p.cost for p in parent_states) / len(parent_states) if parent_states else 0.0
                    print(f"[R={r_val_for_log:3.1f}][MP K-Parent Iter={n_iter:5d}] T={current_temp_t:8.4f} | Cost_avg={avg_cost:8.3f} | Best={global_best_cost:8.3f}")

    except Exception as e:
        print(f"[Error][R={r_val_for_log}] 멀티프로세싱 풀 생성/실행 중 오류: {e}")

    return global_best_chip

# ─────────────────────────────────────────────────────────────
# NEW: final_hybrid_sa 함수 추가
# ─────────────────────────────────────────────────────────────
def final_hybrid_sa(chip_obj, scaler: CostScaler, weights: CostWeights, max_iter=40000, k_workers=None,
                    P_initial=0.8, c_cooling=10, exploitation_depth=10, exploitation_samples=30,
                    exploration_steps=50, reheating_threshold=None, r_val_for_log=0.0):
    """
    최종 미세 조정을 위한 하이브리드 SA.
    """
    if k_workers is None:
        k_workers = max(1, mp.cpu_count() - 2)
    if k_workers == 0: k_workers = 1
    
    global_best_chip = copy.deepcopy(chip_obj)
    global_best_cost = calc_combined_cost(global_best_chip.modules, weights=weights, chip=global_best_chip, scaler=scaler, use_dead_space_in_cost=True)
    
    explorer_chip = copy.deepcopy(global_best_chip)
    explorer_cost = global_best_cost

    uphill_differences = []
    temp_chip_for_t1 = copy.deepcopy(chip_obj)
    for _ in range(50):
        action_name_t1 = select_random_action()
        old_cost_t1 = calc_combined_cost(temp_chip_for_t1.modules, weights=weights, chip=temp_chip_for_t1, scaler=scaler, use_dead_space_in_cost=True)
        temp_chip_for_t1.apply_specific_operation(action_name_t1)
        new_cost_t1 = calc_combined_cost(temp_chip_for_t1.modules, weights=weights, chip=temp_chip_for_t1, scaler=scaler, use_dead_space_in_cost=True)
        delta_e_t1 = new_cost_t1 - old_cost_t1
        if delta_e_t1 > 0:
            uphill_differences.append(delta_e_t1)
        temp_chip_for_t1 = copy.deepcopy(chip_obj)
    
    avg_uphill_delta = sum(uphill_differences) / len(uphill_differences) if uphill_differences else 1.0
    if avg_uphill_delta < 1e-12: avg_uphill_delta = 1.0
    temp_t1_initial = abs(avg_uphill_delta / math.log(P_initial)) if P_initial < 1.0 else 1.0

    last_improvement_iter = 0
    
    try:
        with mp.Pool(processes=k_workers) as pool:
            main_iter_count = 0
            while main_iter_count < max_iter:
                
                if reheating_threshold and (main_iter_count - last_improvement_iter > reheating_threshold):
                    current_temp_t = temp_t1_initial * 0.15
                    last_improvement_iter = main_iter_count
                else:
                    current_temp_t = max((temp_t1_initial * avg_uphill_delta) / ((main_iter_count + 1) * c_cooling), 1e-6)

                worker_tasks = []
                for i in range(k_workers):
                    worker_data = {
                        'worker_id': i, 'chip_state': global_best_chip, 'max_depth': exploitation_depth,
                        'temperature': current_temp_t, 'weights': weights, 'scaler': scaler,
                        'use_ds_in_cost_sa': True, 'sample_moves_num': exploitation_samples
                    }
                    worker_tasks.append(worker_data)
                
                best_cost_before_iter = global_best_cost
                
                try:
                    worker_results = pool.map(worker_sa_depth_search, worker_tasks)
                    for result in worker_results:
                        if result['success'] and result['best_cost'] < global_best_cost:
                            global_best_cost = result['best_cost']
                            global_best_chip = copy.deepcopy(result['best_chip'])
                except Exception as e:
                    print(f"[Warning][R={r_val_for_log}] 3단계 집중 탐색 중 오류 발생: {e}")

                if global_best_cost < best_cost_before_iter:
                    last_improvement_iter = main_iter_count

                for _ in range(exploration_steps):
                    cost_before_explore = explorer_cost
                    
                    chip_after_explore, cost_after_explore = select_best_of_n_moves(
                        explorer_chip, 5, weights, scaler, True
                    )
                    
                    delta_e = cost_after_explore - cost_before_explore
                    if delta_e < 0 or (current_temp_t > 1e-12 and random.random() < math.exp(-abs(delta_e) / current_temp_t)):
                        explorer_chip = chip_after_explore
                        explorer_cost = cost_after_explore
                        
                        if explorer_cost < global_best_cost:
                            global_best_cost = explorer_cost
                            global_best_chip = copy.deepcopy(explorer_chip)
                            last_improvement_iter = main_iter_count

                main_iter_count += (k_workers * exploitation_depth) + exploration_steps

                if main_iter_count % 1000 < 100:
                    print(f"[R={r_val_for_log:3.1f}][Hybrid SA Iter={main_iter_count:6d}] T={current_temp_t:8.4f} | Best Cost={global_best_cost:8.3f}")

    except Exception as e:
        print(f"[Error][R={r_val_for_log}] 3단계 하이브리드 SA 실행 중 오류: {e}")

    return global_best_chip

# ─────────────────────────────────────────────────────────────
# 파일 파싱(Yal, GSRC) 
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
# [수정] 직렬 최적화 실행 함수
# ─────────────────────────────────────────────────────────────
def run_single_optimization_pipeline(r_val, k_val, modules, base_weights):
    """
    하나의 종횡비(R)에 대한 전체 최적화 파이프라인을 실행하는 함수.
    """
    process_id = os.getpid()
    print(f"[R={r_val:.1f}] 최적화 시작 (할당된 워커: {k_val}개)")

    # --- 각 R값에 대해 독립적인 스케일러 초기화 ---
    chip_main = Chip(copy.deepcopy(modules), aspect_ratio=r_val)
    chip_main.calculate_coordinates()
    
    init_aN_raw, init_hN_raw, init_pN_raw, init_dsN_raw = calc_combined_cost(
        chip_main.modules, weights=base_weights, chip=chip_main, return_raw_normalized=True
    )
    scaler = CostScaler()
    scaler.initialize_scales(init_aN_raw, init_hN_raw, init_pN_raw, init_dsN_raw)
    print(f"[R={r_val:.1f}] 독립 스케일러 초기화 완료 (초기 페널티: {init_pN_raw:.2f})")
    
    # [1단계] 넓은 탐색
    print(f"[R={r_val:.1f}] 1단계 K-Parent SA (넓은 탐색) 진행 중...")
    stage1_weights = copy.deepcopy(base_weights)
    stage1_weights.r_dead_space = 1.0
    stage1_weights.r_penalty *= 100
    first_sa_best_chip = multiprocess_k_parent_sa(
        chip_main, scaler=scaler, weights=stage1_weights, max_iter=10000,
        k_parents=k_val, P_initial=0.95, c_cooling=100, sample_moves_num=5,  
        use_ds_in_cost_sa=True, refinement_steps=15, r_val_for_log=r_val
    )
    first_sa_best_chip.plot_b_tree(iteration=f"1st SA Result R{r_val:.1f}", title_suffix="", save_only=True)

    # [2단계] 집중 탐색 및 Pruning
    print(f"[R={r_val:.1f}] 2단계 K-Parent SA (집중 탐색) 진행 중...")
    stage2_weights = copy.deepcopy(base_weights)
    stage2_weights.r_penalty *= 10000  # 페널티 가중치 대폭 상향
    stage2_weights.r_dead_space = 10.0
    second_sa_best_chip = multiprocess_k_parent_sa(
        first_sa_best_chip, scaler=scaler, weights=stage2_weights, max_iter=20000, 
        k_parents=k_val, P_initial=0.95, c_cooling=100, sample_moves_num=10,   
        use_ds_in_cost_sa=True, refinement_steps=25, pruning_start_iter=3000,
        pruning_interval=1000, mutation_strength=5, reheating_threshold=2000, r_val_for_log=r_val
    )
    second_sa_best_chip.plot_b_tree(iteration=f"2nd SA Result R{r_val:.1f}", title_suffix="", save_only=True)
    
    # [3단계] 최종 미세 조정 (하이브리드 SA)
    print(f"[R={r_val:.1f}] 3단계 하이브리드 SA (최종 미세 조정) 진행 중...")
    stage3_weights = copy.deepcopy(base_weights)
    stage3_weights.r_penalty *= 40000  # 페널티 가중치를 매우 높게 설정
    stage3_weights.r_dead_space *= 50 # 데드스페이스 가중치도 함께 상향
    third_sa_best_chip = final_hybrid_sa(
        second_sa_best_chip, scaler=scaler, weights=stage3_weights, max_iter=40000,
        k_workers=k_val, P_initial=0.8, c_cooling=10, exploitation_samples=30,
        reheating_threshold=5000, r_val_for_log=r_val
    ) 
    third_sa_best_chip.plot_b_tree(iteration=f"3rd Hybrid SA Result R{r_val:.1f}", title_suffix="", save_only=True)
    
    # --- 최종 Compaction ---
    third_sa_best_chip.compact_floorplan_final()
    third_sa_best_chip.plot_b_tree(iteration=f"Final Compacted R{r_val:.1f}", title_suffix="", save_only=True) 

    print(f"[R={r_val:.1f}] 최적화 완료.")
    return third_sa_best_chip


# ─────────────────────────────────────────────────────────────
# 5. 메인 실행 
# ─────────────────────────────────────────────────────────────
if __name__=="__main__":
    # 멀티프로세싱을 위한 설정
    try:
        mp.set_start_method('spawn', force=True)
        print(f"[Info] 멀티프로세싱 시작 방법: spawn")
    except RuntimeError:
        print(f"[Info] 멀티프로세싱 시작 방법이 이미 설정됨")
    
    print(f"[Info] 사용 가능한 CPU 코어 수: {mp.cpu_count()}")
    
    # Yal 파일 또는 GSRC 파일 선택 사용
    # --- Yal 예제 ---
    blocks_file="C:/Users/KMT/Desktop/SAF/code/yal/example/hp.yal"
    modules=parse_yal(blocks_file)

    #--- GSRC 예제 ---
    #blocks_file="./example/n100.blocks"
    #nets_file  ="./example/n100.nets"
    #modules=parse_gsrc_blocks(blocks_file)
    #modules=parse_gsrc_nets(nets_file,modules) # nets 정보는 blocks 파싱 후에 추가

    if not modules:
        print("[Error] 로드된 모듈 없음. 종료합니다.") 
        sys.exit(1)

    ASPECT_RATIOS = [1.0, 2.0, 3.0]
    print(f"\n[Info] 다중 종횡비 최적화를 시작합니다. 대상 R: {ASPECT_RATIOS}")

    num_total_workers = max(1, mp.cpu_count() - 2)
    print(f"[Info] 각 종횡비 최적화에 {num_total_workers}개의 워커(코어)를 사용합니다.")
    
    base_weights = CostWeights(w=0.66, r_penalty=1.0, r_dead_space=80.0)
    
    # --- [수정] 직렬 최적화 실행 ---
    start_time = time.time()
    print("\n" + "="*20 + " 직렬 최적화 시작 " + "="*20)
    
    final_chips = []
    for r_val in ASPECT_RATIOS:
        print(f"\n{'='*15} R = {r_val:.1f}에 대한 최적화를 시작합니다. {'='*15}")
        final_chip_for_r = run_single_optimization_pipeline(r_val, num_total_workers, modules, base_weights)
        final_chips.append(final_chip_for_r)
        print(f"\n{'='*15} R = {r_val:.1f}에 대한 최적화가 완료되었습니다. {'='*15}")

    end_time = time.time()
    print("\n" + "="*20 + f" 전체 직렬 최적화 종료 (총 소요 시간: {end_time - start_time:.2f}초) " + "="*20)
    
    # --- 최종 결과 처리 및 비교 ---

    
    print("\n\n" + "="*25 + " 최종 결과 요약 " + "="*25)
    
    final_results_data = []
    # 최종 가중치는 3단계에서 사용한 가중치
    final_weights = copy.deepcopy(base_weights)
    final_weights.r_penalty = 2000
    final_weights.r_dead_space = 100

    # 최종 결과를 출력하기 위한 스케일러 (R=1.0 기준으로 다시 생성하여 일관된 비용 비교)
    final_scaler = CostScaler()
    temp_chip_final = Chip(modules, aspect_ratio=1.0)
    temp_chip_final.calculate_coordinates()
    init_aN, init_hN, init_pN, init_dsN = calc_combined_cost(
        temp_chip_final.modules, weights=base_weights, chip=temp_chip_final, return_raw_normalized=True
    )
    final_scaler.initialize_scales(init_aN, init_hN, init_pN, init_dsN)

    for i, chip in enumerate(final_chips):
        r_val = ASPECT_RATIOS[i]
        final_cost, _, _, _, _ = calc_combined_cost(
            chip.modules, weights=final_weights, chip=chip, scaler=final_scaler,
            use_dead_space_in_cost=True, return_all=True 
        )
        final_w, final_h, final_area = calculate_total_area(chip.modules)
        final_hpwl = calculate_hpwl(chip.modules)
        final_total_mod_area = sum(m.area for m in chip.modules)
        actual_dead_space_area = final_area - final_total_mod_area
        actual_dead_space_percent = (actual_dead_space_area / final_area) * 100 if final_area > 1e-9 else 0.0
        
        results = {
            "R": r_val, "W": final_w, "H": final_h, "Area": final_area,
            "HPWL": final_hpwl, "DeadSpace(%)": actual_dead_space_percent, "Cost": final_cost
        }
        final_results_data.append(results)

        print(f"\n--- 결과 (목표 R = {r_val:.1f}) ---")
        # [수정] 실제 R값 표시를 H/W로 변경
        print(f"최종 경계 상자: W={final_w:.2f}, H={final_h:.2f}, 면적={final_area:.2f} (실제 R: {final_h/final_w if final_w>0 else float('inf'):.2f})")
        print(f"최종 HPWL (절대값)         = {final_hpwl:.2f}")
        print(f"최종 실제 DeadSpace 면적 = {actual_dead_space_area:.2f} ({actual_dead_space_percent:.2f}%)")
        print(f"최종 비용 (w={final_weights.w:.2f}, r_penalty={final_weights.r_penalty:.2f}, r_ds={final_weights.r_dead_space:.2f}) = {final_cost:.3f}")

    print("\n" + "="*65)
    print("\n3가지 종횡비에 대한 플로어플래닝 프로세스 완료.")
    print(f"[Info] 모든 플롯이 PNG 파일로 저장되었습니다.")
    
    show_final_plot = input("\n가장 좋아 보이는 결과(R=1, 2, 3 중 선택)를 화면에 표시하시겠습니까? (1/2/3/n): ")
    if show_final_plot.lower() in ['1', '2', '3']:
        try:
            choice_index = int(show_final_plot) - 1
            chosen_chip = final_chips[choice_index]
            r_val = ASPECT_RATIOS[choice_index]
            # GUI 백엔드로 변경 시도
            matplotlib.use('TkAgg')
            chosen_chip.plot_b_tree(iteration=f"Final Display R{r_val:.1f}", title_suffix="")
            plt.show()
        except Exception as e:
            print(f"[Info] 화면 표시 실패: {e}. PNG 파일을 확인해주세요.")
