# p1,p2 -> io 핀 터미널들인데, test3에선 가져옴, test6에선 안가져오도록 파싱
#test6에선 t1 scale과 sample moves 값이 낮아서 test7에선 이를 수정, 그리고 랜덤배치.
import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import random
import re

# ─────────────────────────────────────────
# 1) Module, BTreeNode, Chip 클래스
# ─────────────────────────────────────────

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
        self.order = None  # 순서 번호

    def rotate(self):
        self.width, self.height = self.height, self.width

    def set_position(self, x: float, y: float, order: int):
        self.x = x
        self.y = y
        self.order = order

    def __str__(self):
        net_list = "\n    ".join(self.net)
        return (f"Module(Name={self.name}, "
                f"W={self.width:.2f}, H={self.height:.2f}, "
                f"Area={self.area:.2f}, Pos=({self.x:.2f}, {self.y:.2f}), "
                f"Order={self.order}, Type={self.type}, Net=\n    {net_list})")


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
    B*-Tree 기반 배치를 관리하는 클래스.
    """
    def __init__(self, modules, default_width=1000, default_height=1000):
        # PARENT 모듈이 없으면 default bound 생성
        self.modules = [m for m in modules if m.type != 'PARENT']
        self.bound = next((m for m in modules if m.type == 'PARENT'), None)
        if not self.bound:
            self.bound = Module(name='DefaultParent',
                                width=default_width,
                                height=default_height,
                                module_type='PARENT')
        # B*-Tree 생성
        self.build_b_tree()

        # Contour
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
        result = []
        queue = [self.root]
        while queue:
            cur = queue.pop(0)
            result.append(cur)
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)
        return result

    def rotate_node(self, node):
        node.module.rotate()

    def move_node(self, node):
        """단순예시: node가 root면 이동X"""
        if node.parent is None:
            return "[Move] It's root, can't move"
        self.remove_node_from_parent(node)
        cands = self.find_possible_parents()
        if not cands:
            return "[Move] No candidate"
        new_parent = random.choice(cands)
        side = self.attach_node(new_parent, node)
        return f"[Move] {node.module.name} -> {new_parent.module.name}({side})"

    def remove_node_from_parent(self, node):
        p = node.parent
        if p and p.left is node:
            p.left = None
        elif p and p.right is node:
            p.right = None
        node.parent = None

    def find_possible_parents(self):
        all_nodes = self.collect_all_nodes()
        return [nd for nd in all_nodes if (nd.left is None or nd.right is None)]

    def attach_node(self, parent, node):
        node.parent = parent
        slots = []
        if parent.left is None:
            slots.append("left")
        if parent.right is None:
            slots.append("right")
        if not slots:
            return "NoSlot"
        chosen = random.choice(slots)
        if chosen=='left':
            parent.left = node
        else:
            parent.right = node
        return chosen

    def swap_nodes(self, A, B):
        A.module, B.module = B.module, A.module

    def apply_random_operation(self):
        ops = ["rotate", "move", "swap"]
        op = random.choice(ops)
        nodes = self.collect_all_nodes()
        if not nodes:
            return ("NoNode", None)

        if op=="rotate":
            sel = random.choice(nodes)
            self.rotate_node(sel)
            msg = f"[Rotate] {sel.module.name}"
        elif op=="move":
            sel = random.choice(nodes)
            msg = self.move_node(sel)
        else:  # swap
            if len(nodes)<2:
                msg = "[Swap] Not enough nodes"
            else:
                A, B = random.sample(nodes, 2)
                self.swap_nodes(A, B)
                msg = f"[Swap] {A.module.name}<->{B.module.name}"

        print(msg)

        # 좌표 초기화 후 재배치
        for nd in nodes:
            nd.module.x = 0
            nd.module.y = 0
            nd.module.order = None
        self.calculate_coordinates()
        return (msg, op)

    # ─────────────────────────────────────────
    # Contour + DFS
    # ─────────────────────────────────────────
    def calculate_coordinates(self):
        self.contour_line = []
        self.max_width = 0
        self.max_height = 0
        all_nds = self.collect_all_nodes()
        for nd in all_nds:
            nd.module.x = 0
            nd.module.y = 0
            nd.module.order = None
        self._dfs_place(self.root, 1, 0, 0)

    def _dfs_place(self, node, order, x_offset, y_offset):
        if not node:
            return order
        mod = node.module
        mod.x = x_offset
        mod.y = y_offset
        mod.order = order

        x1 = x_offset
        x2 = x_offset + mod.width
        top_y = y_offset + mod.height

        self.insert_contour_segment(x1, x2, top_y)
        self.max_width = max(self.max_width, x2)
        self.max_height = max(self.max_height, top_y)

        order+=1

        # 왼쪽
        if node.left:
            lx_start = x_offset + mod.width
            lx_end   = lx_start + node.left.module.width
            ly = self.update_contour(lx_start, lx_end)
            order = self._dfs_place(node.left, order, lx_start, ly)
        # 오른쪽
        if node.right:
            rx_start = x_offset
            rx_end   = rx_start + node.right.module.width
            ry = self.update_contour(rx_start, rx_end)
            order = self._dfs_place(node.right, order, rx_start, ry)

        return order

    def update_contour(self, x1, x2):
        base = 0
        for seg in self.contour_line:
            if not (seg.x2<=x1 or seg.x1>=x2):
                base = max(base, seg.y2)
        return base

    def insert_contour_segment(self, x1, x2, new_top):
        updated=[]
        i=0
        while i<len(self.contour_line):
            seg=self.contour_line[i]
            if seg.x2<=x1 or seg.x1>=x2:
                updated.append(seg)
            else:
                if seg.x1<x1 and seg.x2> x2:
                    updated.append(ContourNode(seg.x1, x1, seg.y2))
                    updated.append(ContourNode(x2, seg.x2, seg.y2))
                elif seg.x1< x1 and seg.x2<= x2:
                    updated.append(ContourNode(seg.x1, x1, seg.y2))
                elif seg.x1>= x1 and seg.x2> x2:
                    updated.append(ContourNode(x2, seg.x2, seg.y2))
            i+=1
        updated.append(ContourNode(x1, x2, new_top))
        updated.sort(key=lambda s:s.x1)
        self.contour_line = updated

    def plot_b_tree(self):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))

        # (1) 물리적 배치
        self._plot_node(self.root, ax1)
        br = plt.Rectangle((0,0), self.bound.width, self.bound.height,
                           edgecolor='red', facecolor='none', lw=2)
        ax1.add_patch(br)
        ax1.set_title("B*-Tree Placement (Contour+DFS)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_xlim(0, self.bound.width+50)
        ax1.set_ylim(0, self.bound.height+50)

        # (2) 트리 구조
        G=nx.DiGraph()
        pos={}
        self._add_tree_edges(self.root, None, depth=0, x_pos=0, G=G, pos=pos)
        nx.draw(G, pos, with_labels=True, node_color="lightblue",
                edge_color="gray", node_size=1200, font_size=9, ax=ax2)
        ax2.set_title("B*-Tree Structure")

        plt.show()

    def _plot_node(self, node, ax):
        if not node:
            return
        mod=node.module
        rect=plt.Rectangle((mod.x,mod.y), mod.width, mod.height,
                           edgecolor='blue', facecolor='lightblue', lw=2)
        ax.add_patch(rect)
        ax.text(mod.x+mod.width/2, mod.y+mod.height/2,
                f"{mod.name}\n#{mod.order}", ha='center',va='center',fontsize=8)
        self._plot_node(node.left, ax)
        self._plot_node(node.right, ax)

    def _add_tree_edges(self, node, parent_name, depth, x_pos, G, pos):
        if not node:
            return x_pos
        node_name=node.module.name
        G.add_node(node_name)
        pos[node_name]=(x_pos, -depth)
        if parent_name:
            G.add_edge(parent_name, node_name)
        x_pos=self._add_tree_edges(node.left, node_name, depth+1, x_pos, G, pos)
        x_pos=self._add_tree_edges(node.right,node_name,depth+1, x_pos+1, G, pos)
        return x_pos

    def randomize_b_tree(self, w=0.5):
        random.shuffle(self.modules)
        self.build_b_tree()
        self.calculate_coordinates()
        return calc_combined_cost(self.modules, w, chip=self)


# ─────────────────────────────────────────
# 2) Cost 계산 함수들
# ─────────────────────────────────────────

def calculate_hpwl(modules):
    net_map={}
    for m in modules:
        for netn in m.net:
            net_map.setdefault(netn,[]).append(m)

    if not net_map:
        return 0

    total=0
    for net,ms in net_map.items():
        minx=min(x.x for x in ms)
        maxx=max(x.x + x.width for x in ms)
        miny=min(x.y for x in ms)
        maxy=max(x.y + x.height for x in ms)
        total+=(maxx-minx)+(maxy-miny)
    return total

def calculate_total_area(modules):
    if not modules:
        return 0,0,0
    minx=min(m.x for m in modules)
    maxx=max(m.x + m.width for m in modules)
    miny=min(m.y for m in modules)
    maxy=max(m.y + m.height for m in modules)
    w=maxx-minx
    h=maxy-miny
    return w,h,(w*h)

def calc_combined_cost(modules, w=0.5, chip=None):
    cost_scale=2000
    if not modules:
        return 0

    base_scale=sum(m.area for m in modules)
    net_area_sum=sum(m.area for m in modules if m.net)
    if net_area_sum==0:
        net_area_sum=base_scale

    import math
    hpwl_scale=2*math.sqrt(net_area_sum)

    _,_,area_bb=calculate_total_area(modules)
    hpwl=calculate_hpwl(modules)
    area_norm=area_bb/base_scale if base_scale>0 else 0
    hpwl_norm=hpwl/hpwl_scale if hpwl_scale>0 else 0
    cost=w*area_norm+(1-w)*hpwl_norm

    if chip:
        for m in modules:
            if (m.x+m.width)>chip.bound.width or (m.y+m.height)>chip.bound.height:
                cost*=1.0
                break
    return cost_scale*cost


# ─────────────────────────────────────────
# 3) FastSA 함수
# ─────────────────────────────────────────

def fast_sa(chip, max_iter=50, P=0.99, c=100, w=0.5, sample_moves=30):
    import copy, math, random

    original_state=copy.deepcopy(chip)
    original_cost=calc_combined_cost(chip.modules, w, chip=chip)

    uphill_diffs=[]
    for _ in range(sample_moves+5):
        msg,op=chip.apply_random_operation()
        new_cost=calc_combined_cost(chip.modules, w, chip=chip)
        delta_e=new_cost - original_cost
        if delta_e>0:
            uphill_diffs.append(delta_e)
        chip=copy.deepcopy(original_state)

    if uphill_diffs:
        delta_avg=sum(uphill_diffs)/len(uphill_diffs)
    else:
        delta_avg=1.0
    if delta_avg<1e-12:
        delta_avg=1.0

    T1_scale=10
    T1=abs(delta_avg/math.log(P))*T1_scale
    print(f"Initial T1 = {T1:.3f}")

    best_chip=copy.deepcopy(chip)
    best_cost=original_cost
    current_cost=original_cost

    for n in range(1, max_iter+1):
        st_cp=copy.deepcopy(chip)
        cost_diffs=[]
        for _ in range(sample_moves):
            msgt, opt=chip.apply_random_operation()
            tmp_c=calc_combined_cost(chip.modules, w, chip=chip)
            cost_diffs.append(abs(tmp_c-current_cost))
            chip=copy.deepcopy(st_cp)
        if cost_diffs:
            delta_cost=max(sum(cost_diffs)/len(cost_diffs),1e-6)
        else:
            delta_cost=1e-6

        old_state=copy.deepcopy(chip)
        old_c=current_cost

        msg2,op2=chip.apply_random_operation()
        new_c=calc_combined_cost(chip.modules, w, chip=chip)
        delta_e=new_c - old_c

        if n==1:
            T=T1
        elif 2<=n<=7:
            T=max((T1*delta_cost)/(n*c),1e-6)
        else:
            T=max((T1*delta_cost)/n,1e-6)

        if delta_e<0:
            current_cost=new_c
            if new_c<best_cost:
                best_cost=new_c
                best_chip=copy.deepcopy(chip)
            accept_str="ACCEPT (better)"
            accept_prob=1.0
        else:
            if T<1e-12:
                accept_prob=0.0
            else:
                accept_prob=math.exp(-abs(delta_cost)/T)

            if random.random()<accept_prob:
                current_cost=new_c
                if new_c<best_cost:
                    best_cost=new_c
                    best_chip=copy.deepcopy(chip)
                accept_str="ACCEPT (worse)"
            else:
                chip=copy.deepcopy(old_state)
                current_cost=old_c
                accept_str="REJECT"

        print(f"[Iter={n:3d}] Op={op2.upper():5s}, T={T:9.5f}, "
              f"deltaCost={delta_cost:9.5f}, Prob={accept_prob:6.4f}, {accept_str}")

    return best_chip


# ─────────────────────────────────────────
# 4) GSRC 파싱 함수들 (Terminal 무시)
# ─────────────────────────────────────────

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

# ─────────────────────────────────────────
# 5) 메인 함수
# ─────────────────────────────────────────

if __name__=="__main__":
    blocks_file = "./example/n300.blocks"
    nets_file   = "./example/n300.nets"

    # (1) GSRC blocks & nets 파싱 (터미널 무시)
    modules = parse_gsrc_blocks(blocks_file)
    modules = parse_gsrc_nets(nets_file, modules)

    # (2) Chip 생성 후, 초기 배치/시각화
    chip = Chip(modules)
    chip.calculate_coordinates()
    chip.plot_b_tree()

    # (3) 초기 배치 스펙
    print("=== 초기 Chip spec ===")
    w_, h_, area_ = calculate_total_area(chip.modules)
    hpwl_ = calculate_hpwl(chip.modules)
    print(f"BoundingBox area = {area_}, HPWL = {hpwl_}")
    init_area = area_
    init_hpwl = hpwl_
    init_cost = calc_combined_cost(chip.modules, w=0.5, chip=chip)

    # FastSA 실행
    ans = input("FastSA로 최적화를 진행하시겠습니까? (y/n): ")
    if ans.lower().startswith('y'):
        best_chip = fast_sa(
            chip,
            max_iter=500000,
            P=0.95,
            c=100,
            w=1,
            sample_moves=4
        )
        print("=== FastSA 종료 결과 ===")
        final_w, final_h, final_area = calculate_total_area(best_chip.modules)
        final_hpwl = calculate_hpwl(best_chip.modules)
        final_cost = calc_combined_cost(best_chip.modules, w=0.5, chip=best_chip)

        # dead space 계산
        total_module_area = sum(m.area for m in best_chip.modules)
        if final_area > 0:
            dead_space_rate = ((final_area - total_module_area) / final_area) * 100
        else:
            dead_space_rate = 0.0

        print(f"초기 BoundingBox area = {init_area}, HPWL = {init_hpwl}")
        print(f"초기 cost = {init_cost:.3f}")
        print(f"BoundingBox area = {final_area}, HPWL = {final_hpwl}")
        print(f"최종 cost = {final_cost:.3f}")
        print(f"Dead space rate = {dead_space_rate:.3f}%")

        best_chip.plot_b_tree()
        plt.show()
