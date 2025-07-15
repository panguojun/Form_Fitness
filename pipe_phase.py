import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from collections import defaultdict
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import NearestNeighbors
from matplotlib.widgets import TextBox
from matplotlib.patches import Rectangle

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 1. 管道系统配置
directions = ['F', 'U', 'B', 'D']  # 前进、上升、后退、下降
n_nodes = 12  # 管道节点数（包含起点和终点）
fixed_start, fixed_end = 'F', 'D'  # 固定起点和终点方向
start_pos = np.array([0, 0])  # 起点坐标
end_pos = np.array([5, -3])   # 终点坐标

# 环境区域定义
class EnvironmentZones:
    def __init__(self):
        self.danger_zones = [
            (2, 1, 4, 3),      # 矩形危险区域1 (x1,y1,x2,y2)
            (-3, -2, -1, 1),   # 矩形危险区域2
            (3, -3, 5, -1),    # 新增危险区域
        ]
        self.obstacles = [
            (0, 3, 1, 4),      # 障碍物1
            (-2, -1, -1, 1),   # 障碍物2
        ]
        self.affinity_zones = [
            (-1, 2, 1, 3),     # 矩形亲和区域1
            (-2, -3, 2, -2),   # 矩形亲和区域2
        ]
        self.safety_margin = 0.8
    
    def point_in_danger_zone(self, x, y):
        for x1, y1, x2, y2 in self.danger_zones:
            if (x1 - self.safety_margin <= x <= x2 + self.safety_margin and 
                y1 - self.safety_margin <= y <= y2 + self.safety_margin):
                return True
        return False
    
    def point_in_obstacle(self, x, y):
        for x1, y1, x2, y2 in self.obstacles:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False
    
    def point_in_affinity_zone(self, x, y):
        for x1, y1, x2, y2 in self.affinity_zones:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

env_zones = EnvironmentZones()

# 核心约束检查
def is_valid_path(path):
    # 检查相邻方向冲突
    for i in range(len(path)-1):
        if (path[i] in ['U', 'D'] and path[i+1] in ['U', 'D'] and path[i] != path[i+1]):
            return False
        if (path[i] in ['F', 'B'] and path[i+1] in ['F', 'B'] and path[i] != path[i+1]):
            return False
    
    # 检查自交
    pos = np.zeros(2)
    visited = set()
    visited.add(tuple(pos))
    for d in path[1:]:
        move = {'F': [1,0], 'B': [-1,0], 'U': [0,1], 'D': [0,-1]}[d]
        pos += move
        tpos = tuple(pos)
        if tpos in visited:
            return False
        visited.add(tpos)
    
    return True

# 2. 能量计算函数
def calculate_energy_components(path):
    components = {
        'total': 0,
        'danger_zone': 0,
        'obstacle': 0,
        'affinity': 0,
        'sharp_turn': 0,
        'direction_change': 0,
        'efficiency': 0,
        'length': 0
    }
    
    pos = np.zeros(2)
    visited = set()
    visited.add(tuple(pos))
    prev_dir = path[0]
    
    for i, d in enumerate(path[1:], 1):
        move = {'F': [1,0], 'B': [-1,0], 'U': [0,1], 'D': [0,-1]}[d]
        pos += move
        visited.add(tuple(pos))
        
        # 急转弯惩罚
        if (prev_dir in ['F', 'B'] and d in ['U', 'D']) or (prev_dir in ['U', 'D'] and d in ['F', 'B']):
            components['sharp_turn'] += 3
            components['total'] += 3
        
        prev_dir = d
        
        # 环境约束检查
        x, y = pos
        if env_zones.point_in_danger_zone(x, y):
            components['danger_zone'] += 150
            components['total'] += 150
        if env_zones.point_in_obstacle(x, y):
            components['obstacle'] += 250
            components['total'] += 250
        if env_zones.point_in_affinity_zone(x, y):
            components['affinity'] += 10
            components['total'] -= 10
    
    # 路径效率评估
    start_to_end = np.linalg.norm(end_pos - start_pos)
    path_length = len(path) - 1
    efficiency_penalty = 10 * (path_length - start_to_end)
    components['efficiency'] = 0.25 * efficiency_penalty
    components['total'] += 0.25 * efficiency_penalty
    
    # 转向惩罚
    dir_changes = sum(1 for i in range(len(path)-1) if path[i] != path[i+1])
    components['direction_change'] = 1 * dir_changes
    components['total'] += 1 * dir_changes
    
    # 路径长度惩罚
    components['length'] = 2 * len(path)
    components['total'] += 2 * len(path)
    
    return components

# 3. 生成初始路径
def generate_initial_path():
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    
    x_dir = 'F' if dx > 0 else 'B'
    y_dir = 'U' if dy > 0 else 'D'
    
    base_path = [x_dir] * abs(dx) + [y_dir] * abs(dy)
    
    path_length = n_nodes - 1
    if len(base_path) > path_length:
        base_path = base_path[:path_length]
    elif len(base_path) < path_length:
        base_path += [base_path[-1]] * (path_length - len(base_path))
    
    full_path = [fixed_start] + base_path[1:-1] + [fixed_end]
    full_path = ''.join(full_path)
    
    if not is_valid_path(full_path):
        raise ValueError("初始路径生成无效")
    
    return full_path

# 4. 路径生成算法
def generate_paths_mcmc(n_samples=1500, temp=0.6):
    current_path = generate_initial_path()
    while not is_valid_path(current_path):
        current_path = generate_initial_path()
    
    current_energy = calculate_energy_components(current_path)['total']
    samples = [current_path]
    energies = [current_energy]
    
    for _ in range(n_samples-1):
        new_path = None
        
        if np.random.rand() < 0.5:  # 交换操作
            idx1, idx2 = np.random.choice(range(1, n_nodes-1), size=2, replace=False)
            temp_path = list(current_path)
            temp_path[idx1], temp_path[idx2] = temp_path[idx2], temp_path[idx1]
            candidate = ''.join(temp_path)
            
            if is_valid_path(candidate):
                new_path = candidate
        
        else:  # 成对插入操作
            pair = ['U', 'D'] if np.random.rand() < 0.5 else ['F', 'B']
            idx1 = np.random.randint(1, n_nodes-3)
            idx2 = np.random.randint(idx1 + 2, n_nodes-1)
            
            temp_path = list(current_path)
            temp_path[idx1] = pair[0]
            temp_path[idx2] = pair[1]
            candidate = ''.join(temp_path)
            
            if is_valid_path(candidate):
                new_path = candidate
        
        if new_path is not None:
            new_energy = calculate_energy_components(new_path)['total']
            
            if (new_energy < current_energy or 
                np.random.rand() < np.exp((current_energy - new_energy) / temp)):
                current_path, current_energy = new_path, new_energy
        
        samples.append(current_path)
        energies.append(current_energy)
    
    return samples, np.array(energies)

# 5. 路径距离度量
def path_distance(path1, path2):
    def get_turns(path):
        return {i for i in range(1, len(path)) if path[i] != path[i-1]}
    turn_dist = len(get_turns(path1).symmetric_difference(get_turns(path2))) * 0.5
    
    def get_shape(path):
        pos = np.zeros(2)
        shape = {tuple(pos)}
        for d in path:
            pos += {'F': [1,0], 'B': [-1,0], 'U': [0,1], 'D': [0,-1]}[d]
            shape.add(tuple(pos))
        return shape
    shape_dist = len(get_shape(path1).symmetric_difference(get_shape(path2))) * 0.8
    
    hist_dist = sum(abs(path1.count(d) - path2.count(d)) for d in directions) * 0.3
    
    return turn_dist + shape_dist + hist_dist

# 6. 改进的路径可视化函数
def plot_path(path, ax=None, title=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    pos = np.zeros(2)
    x, y = [pos[0]], [pos[1]]
    
    # 绘制环境区域
    for i, (x1, y1, x2, y2) in enumerate(env_zones.danger_zones):
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        facecolor='red', alpha=0.2, label=f'危险区域{i+1}')
        ax.add_patch(rect)
    
    for i, (x1, y1, x2, y2) in enumerate(env_zones.affinity_zones):
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        facecolor='green', alpha=0.2, label=f'亲和区域{i+1}')
        ax.add_patch(rect)
    
    # 绘制障碍物
    for (x1, y1, x2, y2) in env_zones.obstacles:
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        facecolor='black', alpha=0.5, label='障碍物')
        ax.add_patch(rect)
    
    # 绘制路径
    for d in path:
        move = {'F': [1,0], 'B': [-1,0], 'U': [0,1], 'D': [0,-1]}[d]
        pos += move
        x.append(pos[0])
        y.append(pos[1])
    
    ax.plot(x, y, 'b-o', linewidth=2, markersize=6, label='管道')
    ax.scatter(x[0], y[0], c='green', s=150, marker='s', label='起点')
    ax.scatter(x[-1], y[-1], c='red', s=150, marker='s', label='终点')
    
    # 设置坐标轴范围
    all_x = x + [zone[i] for zone in env_zones.danger_zones + env_zones.affinity_zones + env_zones.obstacles for i in (0, 2)]
    all_y = y + [zone[i] for zone in env_zones.danger_zones + env_zones.affinity_zones + env_zones.obstacles for i in (1, 3)]
    ax.set_xlim(min(all_x)-1, max(all_x)+1)
    ax.set_ylim(min(all_y)-1, max(all_y)+1)
    
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_title(title or f'管道路径: {path}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 合并重复的图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    ax.set_aspect('equal')
    if fig is not None:
        fig.tight_layout()
    
    if ax is None:
        plt.show()

# 7. 基于邻居生成混合路径
def generate_hybrid_path(neighbor_paths, distances):
    path_length = len(neighbor_paths[0])
    weights = 1 / (np.array(distances) + 1e-6)
    weights /= weights.sum()
    
    direction_probs = []
    for i in range(path_length):
        dir_counts = defaultdict(float)
        for j, path in enumerate(neighbor_paths):
            if i < len(path):
                dir_counts[path[i]] += weights[j]
        direction_probs.append(dir_counts)
    
    new_path = []
    for i in range(path_length):
        if i == 0:
            new_path.append(fixed_start)
        elif i == path_length - 1:
            new_path.append(fixed_end)
        else:
            probs = list(direction_probs[i].values())
            dirs = list(direction_probs[i].keys())
            if not probs:
                new_path.append(np.random.choice(directions))
            else:
                chosen_dir = np.random.choice(dirs, p=np.array(probs)/sum(probs))
                new_path.append(chosen_dir)
    
    if not is_valid_path(''.join(new_path)):
        for i in range(len(new_path)-1):
            current = new_path[i]
            next_dir = new_path[i+1]
            if (current in ['U', 'D'] and next_dir in ['U', 'D'] and current != next_dir):
                new_path[i+1] = np.random.choice(['F', 'B'])
            elif (current in ['F', 'B'] and next_dir in ['F', 'B'] and current != next_dir):
                new_path[i+1] = np.random.choice(['U', 'D'])
        
        if not is_valid_path(''.join(new_path)):
            return neighbor_paths[np.argmin(distances)]
    
    return ''.join(new_path)

# 8. 主程序（改进可视化布局）
def main():
    print("生成多样化路径样本...")
    paths, energies = generate_paths_mcmc(n_samples=1500, temp=0.6)
    
    valid_mask = [is_valid_path(p) for p in paths]
    paths = [p for p, v in zip(paths, valid_mask) if v]
    energies = [e for e, v in zip(energies, valid_mask) if v]
    if len(paths) < 50:
        raise ValueError("有效路径样本不足")
    
    sample_size = min(200, len(paths))
    sample_idx = sorted(np.random.choice(len(paths), size=sample_size, replace=False),
                       key=lambda x: energies[x])
    sample_paths = [paths[i] for i in sample_idx]
    sample_energies = [energies[i] for i in sample_idx]
    
    print("计算路径距离矩阵...")
    dist_matrix = np.zeros((sample_size, sample_size))
    for i, j in combinations(range(sample_size), 2):
        dist = path_distance(sample_paths[i], sample_paths[j])
        dist_matrix[i,j] = dist_matrix[j,i] = dist
    
    print("执行t-SNE降维...")
    tsne = TSNE(n_components=2, metric="precomputed", 
               init="random", random_state=42,
               perplexity=min(30, sample_size//4),
               n_iter=2000)
    sample_coords = tsne.fit_transform(dist_matrix)
    
    # 创建更大的可视化界面
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2], height_ratios=[1, 1])
    
    # 主图区域（能量景观）
    ax = fig.add_subplot(gs[:, 0])
    
    # 右侧区域分为上下两部分
    ax_info = fig.add_subplot(gs[0, 1:])
    ax_info.set_axis_off()
    ax_path = fig.add_subplot(gs[1, 1])
    ax_energy = fig.add_subplot(gs[1, 2])
    
    # 初始化文本框
    text_box = TextBox(ax_info, '', initial='点击图中任意位置生成路径...', 
                      label_pad=0.1, color='lightgray')
    
    # 绘制能量景观
    xi = np.linspace(sample_coords[:,0].min()-1, sample_coords[:,0].max()+1, 200)
    yi = np.linspace(sample_coords[:,1].min()-1, sample_coords[:,1].max()+1, 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((sample_coords[:,0], sample_coords[:,1]), sample_energies, 
                 (xi, yi), method='cubic')
    contour = ax.contourf(xi, yi, zi, levels=20, cmap='viridis', alpha=0.8)
    
    sc = ax.scatter(sample_coords[:,0], sample_coords[:,1], c=sample_energies,
                   cmap='viridis', edgecolors='k', linewidths=0.5, s=60)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_label('路径能量', fontsize=12)
    
    # 标记最优和最差路径
    min_idx = np.argmin(sample_energies)
    max_idx = np.argmax(sample_energies)
    ax.scatter(sample_coords[min_idx,0], sample_coords[min_idx,1], c='lime',
              s=200, marker='o', edgecolors='k', label='最优路径')
    ax.scatter(sample_coords[max_idx,0], sample_coords[max_idx,1], c='red',
              s=200, marker='X', edgecolors='k', label='最差路径')
    ax.legend()
    
    ax.set_title('路径能量景观（点击生成新路径）', fontsize=14, pad=20)
    ax.set_xlabel('t-SNE维度1', fontsize=12)
    ax.set_ylabel('t-SNE维度2', fontsize=12)
    ax.grid(alpha=0.2)
    
    # 点击事件处理
    def onclick(event):
        if event.inaxes != ax:
            return
        
        # 找到最近的5个邻居路径
        click_pos = np.array([[event.xdata, event.ydata]])
        nbrs = NearestNeighbors(n_neighbors=5).fit(sample_coords)
        distances, indices = nbrs.kneighbors(click_pos)
        neighbor_paths = [sample_paths[i] for i in indices[0]]
        
        # 生成混合路径
        new_path = generate_hybrid_path(neighbor_paths, distances[0])
        energy_components = calculate_energy_components(new_path)
        
        # 标记生成位置
        for artist in ax.collections:
            if hasattr(artist, '_generated'):
                artist.remove()
        marker = ax.scatter(click_pos[0,0], click_pos[0,1], 
                          s=120, facecolors='none', 
                          edgecolors='cyan', linewidths=2, label='生成位置')
        marker._generated = True
        ax.legend()
        
        # 更新信息文本
        detail_text = (
            f"生成路径: {new_path}\n"
            f"总能量: {energy_components['total']:.1f}\n"
            "------------------------\n"
            f"危险区域惩罚: {energy_components['danger_zone']:.1f}\n"
            f"障碍物惩罚: {energy_components['obstacle']:.1f}\n"
            f"亲和区域奖励: {energy_components['affinity']:.1f}\n"
            f"急转弯惩罚: {energy_components['sharp_turn']:.1f}\n"
            f"方向变化惩罚: {energy_components['direction_change']:.1f}\n"
            f"效率惩罚: {energy_components['efficiency']:.1f}\n"
            f"长度惩罚: {energy_components['length']:.1f}"
        )
        text_box.set_val(detail_text)
        
        # 更新路径预览
        ax_path.clear()
        plot_path(new_path, ax=ax_path, title=f'路径预览 (能量: {energy_components["total"]:.1f})', fig=fig)
        
        # 更新能量组成分析
        ax_energy.clear()
        components = {
            '危险区域': energy_components['danger_zone'],
            '障碍物': energy_components['obstacle'],
            '亲和区域': -energy_components['affinity'],
            '急转弯': energy_components['sharp_turn'],
            '方向变化': energy_components['direction_change'],
            '效率': energy_components['efficiency'],
            '长度': energy_components['length']
        }
        colors = ['red' if v > 0 else 'green' for v in components.values()]
        bars = ax_energy.barh(list(components.keys()), list(components.values()), 
                             color=colors, alpha=0.6)
        for bar in bars:
            width = bar.get_width()
            ax_energy.text(width + 0.1 if width > 0 else width - 0.1, 
                         bar.get_y() + bar.get_height()/2, 
                         f"{width:.1f}", ha='left' if width >0 else 'right', va='center', fontsize=8)
        ax_energy.set_title('能量组成分析', fontsize=12)
        ax_energy.grid(True, alpha=0.2)
        
        plt.draw()
    
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
