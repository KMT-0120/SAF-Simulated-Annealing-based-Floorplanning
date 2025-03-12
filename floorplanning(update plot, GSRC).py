#해당 주석 문단은 지우지말것(특히 GPT 사용시, 지워지는지 또한 확인한 후 다시 추가), GPT 돌릴떄 함수 이름, 변수명 변경하지 말것
#3/8 iter 5000번마다 plot, SA종료 후 T 변화 추이 그래프 plot 추가
# 사용하지 않는 과거 코드 삭제(adjust_for_overlaps, check_overlap, is_within_bounds)
# plot관련해서 노드 크기/폰트 크기 조정, 노드 간격 조정
# c factor 조정

#3/9 GSRC 적용, 초기해 leftchain -> random으로 변경

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
    """
    B*-Tree (Left-Chain). 부모와 자식 간 +10 오프셋.
    """
    def __init__(self, modules, default_width=1000, default_height=1000):
        self.modules = [m for m in modules if m.type != 'PARENT']
        self.bound = next((m for m in modules if m.type == 'PARENT'), None)
        if not self.bound:
            self.bound = Module(
                name='DefaultParent',
                width=default_width,
                height=default_height,
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
            # 트리 전체에서 '비어있는 left/right'를 랜덤 선택해 붙이기
            cand_nodes = self.find_possible_parents()
            chosen_parent = random.choice(cand_nodes)
            side = self.attach_node(chosen_parent, node)

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
    # 연산들
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

    def apply_random_operation(self):
        ops=["rotate","move","swap"]
        op=random.choice(ops)
        all_nodes=self.collect_all_nodes()
        if not all_nodes:
            print("[Warn] No nodes in the tree.")
            return ("NoNode",None)

        if op=="rotate":
            chosen=random.choice(all_nodes)
            self.rotate_node(chosen)
            msg=f"[Op1: Rotate] Rotated {chosen.module.name}"
        elif op=="move":
            chosen=random.choice(all_nodes)
            msg=self.move_node(chosen)
        else: # swap
            if len(all_nodes)<2:
                msg="[Op3: Swap] Not enough nodes to swap."
            else:
                nodeA,nodeB=random.sample(all_nodes,2)
                self.swap_nodes(nodeA,nodeB)
                msg=f"[Op3: Swap] Swapped {nodeA.module.name} <-> {nodeB.module.name}"

        print(msg)
        # 좌표 초기화 & 재계산
        for nd in all_nodes:
            nd.module.x=0
            nd.module.y=0
            nd.module.order=None
        self.calculate_coordinates()
        return (msg,op)

    def randomize_b_tree(self,w=0.5):
        random.shuffle(self.modules)
        self.build_b_tree()
        self.calculate_coordinates()
        return calc_combined_cost(self.modules,w,chip=self)

    # ─────────────────────────────────────────
    # 좌표 배치 (Contour + DFS, +10)
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

        # 왼쪽 => x+width+10
        if node.left:
            lx=node.module.x+node.module.width+10
            rx=lx+node.left.module.width
            ly=self.update_contour(lx,rx)
            order=self._dfs_place_node(node.left,order,lx,ly)

        # 오른쪽 => x+10
        if node.right:
            rxs=node.module.x+10
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

    # plot( iteration=None ): 더 넓게 노드 배치 + 폰트/노드 크기 조정
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
        # (배율 확대)
        pos[node_name]=(x_pos*2, -depth*2)

        G.add_node(node_name,label=node.module.name)
        if parent_name:
            G.add_edge(parent_name,node_name)

        # 왼쪽
        x_pos=self.add_edges(node.left,node_name,depth+1,x_pos,G,pos)
        # 오른쪽(간격 +2)
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
        max_x=max(mm.x+mm.width for mm in mods)
        min_y=min(mm.y for mm in mods)
        max_y=max(mm.y+mm.height for mm in mods)
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

def calc_combined_cost(modules,w=0.5,chip=None):
    cost_scale=2000
    if not modules:
        return 0.0
    base_scale=sum(m.area for m in modules)
    net_area_sum=sum(m.area for m in modules if m.net)
    if net_area_sum==0:
        net_area_sum=base_scale
    hpwl_scale=2*math.sqrt(net_area_sum)

    _,_,area_bb=calculate_total_area(modules)
    hpwl=calculate_hpwl(modules)
    area_norm=area_bb/base_scale if base_scale else 0
    hpwl_norm=hpwl/hpwl_scale if hpwl_scale>0 else 0
    cost=w*area_norm+(1-w)*hpwl_norm

    if chip is not None:
        for m in modules:
            if (m.x+m.width)>chip.bound.width or (m.y+m.height)>chip.bound.height:
                cost*=1
                break
    return cost*cost_scale


# ─────────────────────────────────────────────────────────────
# 3. FastSA 함수 (서브트리 로컬 SA 제거)
# ─────────────────────────────────────────────

def fast_sa(chip, max_iter=50, P=0.99, c=100, w=0.5, sample_moves=10):
    import copy, math, random
    T1_scale_factor=3.0
    orig_st=copy.deepcopy(chip)
    orig_cost=calc_combined_cost(chip.modules,w,chip=chip)
    up_diffs=[]
    for _ in range(sample_moves+10):
        msg,op=chip.apply_random_operation()
        new_c=calc_combined_cost(chip.modules,w,chip=chip)
        dE=new_c-orig_cost
        if dE>0:
            up_diffs.append(dE)
        chip=copy.deepcopy(orig_st)

    if up_diffs:
        delta_avg=sum(up_diffs)/len(up_diffs)
    else:
        delta_avg=1.0
    if delta_avg<1e-12:
        delta_avg=1.0
    T1=abs(delta_avg / math.log(P)) * T1_scale_factor
    print(f"Initial T1={T1:.3f}")

    best_chip=copy.deepcopy(chip)
    best_cost=orig_cost
    cur_cost=orig_cost

    temps=[]
    for n in range(1,max_iter+1):
        st_cp=copy.deepcopy(chip)
        cost_diffs=[]
        for _ in range(sample_moves):
            msg_tmp,op_tmp=chip.apply_random_operation()
            tmp_cost=calc_combined_cost(chip.modules,w,chip=chip)
            cost_diffs.append(abs(tmp_cost-cur_cost))
            chip=copy.deepcopy(st_cp)
        if cost_diffs:
            delta_cost=max(sum(cost_diffs)/len(cost_diffs),1e-6)
        else:
            delta_cost=1e-6

        if n==1:
            T=T1
        elif 2<=n<=400:
            T=max((T1*delta_cost)/(n*c),1e-6)
        else:
            T=max((T1*delta_cost)/n,1e-6)

        temps.append(T)

        old_chip=copy.deepcopy(chip)
        oldc=cur_cost

        msg3,op3=chip.apply_random_operation()
        new_c=calc_combined_cost(chip.modules,w,chip=chip)
        dE=new_c - oldc

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
                acc_prob=math.exp(-abs(delta_cost)/T)
            if random.random()<acc_prob:
                cur_cost=new_c
                if new_c<best_cost:
                    best_cost=new_c
                    best_chip=copy.deepcopy(chip)
                acc_str="ACCEPT (worse)"
            else:
                chip=copy.deepcopy(old_chip)
                cur_cost=oldc
                acc_str="REJECT"

        print(f"[Iter={n:3d}] Op={op3.upper():5s}, "
              f"T={T:9.5f}, "
              f"deltaCost={delta_cost:9.5f}, "
              f"Prob={acc_prob:6.4f}, {acc_str}")

        # n%5000==0 -> 배치도 plot
        if n%10000==0:
            chip.plot_b_tree(iteration=n)
            plt.show()

    # 최종 온도 그래프
    plt.figure()
    plt.plot(range(1,max_iter+1),temps)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.title("Temperature vs. Iteration")
    plt.show()

    return best_chip


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
    """
    Terminal( pX )은 무시하고, sbX 블록만 파싱
    """
    modules=[]
    pattern=re.compile(r"^(sb\d+)\s+\S+\s+(\d+)\s+(.*)$")
    with open(blocks_file,'r') as f:
        lines=f.readlines()

    for line in lines:
        line=line.strip()
        if not line or line.startswith('#') or line.startswith('UCSC'):
            continue

        # 만약 p1 terminal 나오면 => 그냥 skip
        if line.startswith('p') and 'terminal' in line:
            continue

        match=pattern.match(line)
        if match:
            blk_name=match.group(1)    # sb0, sb1 ...
            coords=match.group(3)     # (0,0) ...
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
    """
    pX 등 terminal은 이미 parse_gsrc_blocks에서 안 넣었으므로,
    name_map에 없는 pin_name => 무시됨
    """
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
                    # pX 는 name_map에 없으므로, 자동 무시
                    if pin_name in name_map:
                        pins.append(pin_name)
            for pn in pins:
                name_map[pn].net.append(net_name)
        i+=1
    return modules



# ─────────────────────────────────────────────────────────────
# 5. 메인 실행 예시 (Left-Chain + 노드 간격/크기 조정, 서브트리 SA 제거)
# ─────────────────────────────────────────────────────────────

if __name__=="__main__":
    # (1) Yal 파일 파싱
    #yal_file="./example/ami49.yal"
    #modules=parse_yal(yal_file)

    # (2) GSRC blocks & nets 파싱 (터미널 무시)
    blocks_file = "./example/n300.blocks"
    nets_file   = "./example/n300.nets"
    modules = parse_gsrc_blocks(blocks_file)
    modules = parse_gsrc_nets(nets_file, modules)


    chip=Chip(modules)
    chip.calculate_coordinates()
    chip.plot_b_tree()

    w_,h_,area_=calculate_total_area(chip.modules)
    hpwl_=calculate_hpwl(chip.modules)
    init_cost=calc_combined_cost(chip.modules,w=0.5,chip=chip)
    print("=== 초기 Chip (Left-Chain) ===")
    print(f"BoundingBox area={area_}, HPWL={hpwl_}, cost={init_cost:.3f}")

    ans=input("FastSA로 최적화를 진행?(y/n): ")
    if ans.lower().startswith('y'):
        best_chip=fast_sa(
            chip,
            max_iter=100000,
            P=0.95,
            c=100,
            w=1,
            sample_moves=4
        )
        final_w,final_h,final_area=calculate_total_area(best_chip.modules)
        final_hpwl=calculate_hpwl(best_chip.modules)
        final_cost=calc_combined_cost(best_chip.modules,w=0.5,chip=best_chip)

        total_module_area=sum(m.area for m in best_chip.modules)
        if final_area>0:
            ds=((final_area-total_module_area)/final_area)*100
        else:
            ds=0.0

        print("=== FastSA 종료 ===")
        print(f"초기 cost={init_cost:.3f}")
        print(f"SA 최종 area={final_area}, HPWL={final_hpwl}, cost={final_cost:.3f}")
        print(f"DeadSpaceRate={ds:.3f}%")

        best_chip.plot_b_tree()
        plt.show()
