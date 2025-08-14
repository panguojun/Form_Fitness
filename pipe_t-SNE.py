import numpy as np
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class ProjectionEngine:
    """
    高维路径数据到二维空间的投影引擎
    
    功能:
    1. 计算路径之间的距离矩阵
    2. 使用t-SNE进行降维投影
    3. 提供交互式查询接口
    4. 生成能量景观可视化数据
    """
    
    def __init__(self, paths, energies, distance_metric=None, tsne_params=None):
        self.paths = paths
        self.energies = np.array(energies)
        self.distance_metric = distance_metric or self.default_distance_metric
        self.tsne_params = tsne_params or {
            'n_components': 2,
            'metric': "precomputed",
            'init': "random",
            'random_state': 42,
            'perplexity': 30,
            'n_iter': 2000
        }
        self.coords = None
        self.dist_matrix = None
        
    @staticmethod
    def default_distance_metric(path1, path2):
        """默认路径距离度量方法"""
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
        hist_dist = sum(abs(path1.count(d) - path2.count(d)) for d in ['F', 'U', 'B', 'D']) * 0.3
        
        return turn_dist + shape_dist + hist_dist
    
    def compute_distance_matrix(self):
        """计算路径之间的距离矩阵"""
        n = len(self.paths)
        self.dist_matrix = np.zeros((n, n))
        
        for i, j in combinations(range(n), 2):
            dist = self.distance_metric(self.paths[i], self.paths[j])
            self.dist_matrix[i,j] = self.dist_matrix[j,i] = dist
        
        return self.dist_matrix
    
    def project_to_2d(self):
        """使用t-SNE将路径投影到二维空间"""
        if self.dist_matrix is None:
            self.compute_distance_matrix()
        
        # 动态调整perplexity
        effective_perplexity = min(self.tsne_params['perplexity'], len(self.paths)//4)
        tsne = TSNE(**{**self.tsne_params, 'perplexity': effective_perplexity})
        
        self.coords = tsne.fit_transform(self.dist_matrix)
        return self.coords
    
    def get_energy_landscape(self, grid_size=200):
        """
        生成能量景观网格数据
        返回:
        - xi, yi: 网格坐标
        - zi: 网格上的能量值
        """
        if self.coords is None:
            self.project_to_2d()
        
        # 创建插值网格
        xi = np.linspace(self.coords[:,0].min()-1, self.coords[:,0].max()+1, grid_size)
        yi = np.linspace(self.coords[:,1].min()-1, self.coords[:,1].max()+1, grid_size)
        xi, yi = np.meshgrid(xi, yi)
        
        # 使用三次插值
        zi = griddata((self.coords[:,0], self.coords[:,1]), self.energies,
                     (xi, yi), method='cubic')
        
        return xi, yi, zi
    
    def find_nearest_neighbors(self, point, n_neighbors=5):
        """
        在投影空间中找到最近的邻居路径
        参数:
        - point: 二维坐标点
        - n_neighbors: 要查找的邻居数量
        返回:
        - indices: 邻居索引
        - distances: 邻居距离
        - paths: 邻居路径
        - energies: 邻居能量
        """
        if self.coords is None:
            self.project_to_2d()
        
        point = np.array(point).reshape(1, -1)
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(self.paths)))
        nbrs.fit(self.coords)
        distances, indices = nbrs.kneighbors(point)
        
        return (
            indices[0],
            distances[0],
            [self.paths[i] for i in indices[0]],
            [self.energies[i] for i in indices[0]]
        )
    
    def get_extreme_paths(self):
        """获取最优和最差路径"""
        if len(self.energies) == 0:
            return None, None
        
        min_idx = np.argmin(self.energies)
        max_idx = np.argmax(self.energies)
        return self.paths[min_idx], self.paths[max_idx]

def generate_test_paths(n_samples=100):
    """生成测试路径"""
    directions = ['F', 'U', 'B', 'D']
    paths = []
    energies = []
    
    for _ in range(n_samples):
        length = np.random.randint(5, 15)
        path = [np.random.choice(directions) for _ in range(length)]
        paths.append(''.join(path))
        energies.append(np.random.uniform(0, 100))
    
    return paths, energies

def test_projection_engine():
    """测试投影引擎"""
    print("=== 开始测试投影引擎 ===")
    
    # 1. 生成测试数据
    print("生成测试路径...")
    test_paths, test_energies = generate_test_paths(n_samples=50)
    
    # 2. 创建投影引擎实例
    print("初始化投影引擎...")
    projector = ProjectionEngine(test_paths, test_energies)
    
    # 3. 计算距离矩阵
    print("计算距离矩阵...")
    dist_matrix = projector.compute_distance_matrix()
    print(f"距离矩阵形状: {dist_matrix.shape}")
    
    # 4. 执行降维投影
    print("执行t-SNE降维...")
    coords = projector.project_to_2d()
    print(f"投影坐标形状: {coords.shape}")
    
    # 5. 获取能量景观
    print("生成能量景观...")
    xi, yi, zi = projector.get_energy_landscape()
    print(f"网格数据 - xi: {xi.shape}, yi: {yi.shape}, zi: {zi.shape}")
    
    # 6. 测试邻居查找
    print("测试邻居查找...")
    test_point = [0, 0]  # 测试点
    indices, dists, neighbor_paths, neighbor_energies = projector.find_nearest_neighbors(test_point)
    print(f"找到 {len(indices)} 个邻居:")
    for i, (idx, dist, path, energy) in enumerate(zip(indices, dists, neighbor_paths, neighbor_energies)):
        print(f"邻居 {i+1}: 索引={idx}, 距离={dist:.2f}, 路径='{path}', 能量={energy:.2f}")
    
    # 7. 获取极值路径
    min_path, max_path = projector.get_extreme_paths()
    print(f"\n最优路径: '{min_path}' (能量={min(test_energies):.2f})")
    print(f"最差路径: '{max_path}' (能量={max(test_energies):.2f})")
    
    # 8. 可视化结果
    print("\n绘制投影结果...")
    plt.figure(figsize=(10, 8))
    
    # 绘制能量景观
    plt.contourf(xi, yi, zi, levels=15, cmap='viridis', alpha=0.6)
    
    # 绘制所有路径点
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=test_energies, 
                         cmap='viridis', s=50, edgecolors='k', linewidths=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('路径能量值', fontsize=12)
    
    # 标记测试点和邻居
    plt.scatter(test_point[0], test_point[1], c='red', s=150, 
               marker='x', linewidths=2, label='测试点')
    plt.scatter(coords[indices, 0], coords[indices, 1], 
               c='cyan', s=100, edgecolors='k', linewidths=1, label='邻居路径')
    
    plt.title('路径空间投影与能量分布', fontsize=14)
    plt.xlabel('t-SNE 维度 1', fontsize=12)
    plt.ylabel('t-SNE 维度 2', fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 测试完成 ===")

if __name__ == "__main__":
    # 检查系统可用字体
    print("系统可用字体:", matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        test_projection_engine()
    except:
        # 如果SimHei不可用，尝试其他字体
        print("SimHei字体不可用，尝试使用其他中文字体...")
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            test_projection_engine()
        except:
            print("Microsoft YaHei字体也不可用，尝试使用Arial Unicode MS...")
            try:
                plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
                test_projection_engine()
            except:
                print("无法找到合适的中文字体，将使用英文显示")
                plt.rcParams['font.sans-serif'] = ['Arial']
                test_projection_engine()