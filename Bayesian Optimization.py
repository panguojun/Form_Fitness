import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import threading
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from datetime import datetime
from gcu_api import DoPHG, get_grid_aabb, cell

# 全局超时控制
class TimeoutGuard:
    def __init__(self, timeout):
        self.timeout = timeout
        self.timed_out = False
        self.thread = None

    def __enter__(self):
        self.timed_out = False
        self.thread = threading.Timer(self.timeout, self.set_timeout)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread.cancel()
        return self.timed_out

    def set_timeout(self):
        self.timed_out = True
        print(f"Operation timed out ({self.timeout} seconds), force termination")

# 转换numpy类型为Python原生类型
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

# 核心参数
SCALER = 40.0  # 缩放因子（厘米→分米）
AIR_DENSITY = 1.225  # kg/m3
MAX_VELOCITY = 30  # m/s
MAX_ITERATIONS = 50  # 最大迭代次数
MAX_GRID_CELLS = 1e7  # 最大网格单元数
GRID_UNIT_SIZE = 0.3  # 网格单位（分米）
OPERATION_TIMEOUT = 15  # 操作超时时间（秒）

class WingEfficiencyOptimizer:
    def __init__(self):
        self.best_params = None
        self.best_ratio = 0  # 最优升力/面积比
        self.best_lift = 0
        self.best_area = 0
        self.iteration = 0
        self.history = []
        self.start_time = datetime.now()
        self.last_successful = None
        
        # 参数范围（确保width1 > width2）
        self.dimensions = [
            Real(name='width1', low=2.0, high=5.0),    # 根弦（2-5分米）
            Real(name='width2', low=0.5, high=2.5),    # 梢弦（0.5-2.5分米）
            Real(name='length', low=15.0, high=30.0),  # 翼展（1.5-3米）
            Real(name='angle1', low=5.0, high=20.0),   # 根角（5-20度）
            Real(name='angle2', low=0.0, high=15.0)    # 梢角（0-15度）
        ]
        self.param_names = [dim.name for dim in self.dimensions]
        
        self.failed_attempts = 0
        self.max_failures = 5
        
        # 新增参数停滞监控
        self.stall_count = 0  # 记录最优解停滞次数
        self.max_stall = 8    # 连续停滞8次则调整参数范围
        
        # 优化曲线数据
        self.valid_iterations = []  # 只记录有效迭代
        self.valid_ratios = []      # 有效升力/面积比
        self.best_ratio_history = [] # 最优比历史

    def calculate_wing_area(self, params_dict):
        """计算机翼面积（梯形面积公式：(根弦+梢弦)/2 * 翼展）"""
        width1 = params_dict['width1']
        width2 = params_dict['width2']
        length = params_dict['length']
        # 转换为平方米（1分米=0.1米）
        area = ((width1 + width2) / 2) * length * (0.1 ** 2)
        return max(area, 0.01)  # 避免面积为0

    def enforce_constraints(self, params_dict):
        """强化物理约束，确保width1 > width2"""
        # 严格保证根弦 > 梢弦（至少大0.3分米，改为更严格的动态约束）
        min_width_diff = 0.2 + 0.05 * params_dict['width1']  # 根弦越大，最小差值越大
        if params_dict['width2'] >= params_dict['width1'] - min_width_diff:
            params_dict['width2'] = params_dict['width1'] - min_width_diff

        # 梢角 ≤ 根角，且增加角度差约束
        max_angle_diff = 0.2 * params_dict['angle1']  # 根角越大，梢角最大差值越大
        params_dict['angle2'] = min(params_dict['angle2'], params_dict['angle1'] - 0.5)  
        params_dict['angle2'] = max(params_dict['angle2'], 0.0)  # 确保非负
        return params_dict

    def pre_check_grid_size(self, params_dict):
        """预检查网格规模，避免生成过大网格"""
        # 更精准的网格估算公式（结合所有参数）
        estimated_grid = (params_dict['length'] * params_dict['width1'] * 
                          (1 + params_dict['angle1']/30) *  # 角度对网格的影响
                          (1 + params_dict['angle2']/20) *  # 梢角修正
                          12)  # 经验系数微调
        if estimated_grid > MAX_GRID_CELLS * 0.8:  # 预留20%缓冲
            print(f"Pre-check: Grid too large (estimated {estimated_grid:.0f} cells), skip params")
            return False
        return True

    def safe_generate_model(self, params, filename="temp_wing"):
        """安全生成模型，带超时控制"""
        try:
            params_dict = {k: convert_numpy_types(v) for k, v in zip(self.param_names, params)}
            # 强制约束
            params_dict = self.enforce_constraints(params_dict)
            
            # 预检查网格规模
            if not self.pre_check_grid_size(params_dict):
                return False, None
            
            # 生成模型脚本
            script = f"""
            {{
            c11{{x:-{params_dict['width1']};rz:0;z:-{params_dict['length']}}}
            c12{{x:{params_dict['width1']};rz:{params_dict['angle1']};z:-{params_dict['length']}}}
            c21{{x:-{params_dict['width2']};rz:-{params_dict['angle2']}}}
            c22{{x:{params_dict['width2']};rz:{params_dict['angle2']}}}
            cv1{{md:ccurve lerpTX c11 c12;rz:15;ts:5}}
            cv2{{md:ccurve lerpTX c21 c22;ts:5}}
            f1{{md:face lerp cv1 cv2}}
            }}setup();draw(f1);
            savesmb("{filename}.sm");
            grid_sm_test('{filename}.sm');
            }}
            """
            
            # 超时控制
            with TimeoutGuard(OPERATION_TIMEOUT) as guard:
                DoPHG(script)  # 实际运行时会调用外部API
                if guard.timed_out:
                    raise TimeoutError("Model generation timed out")
            
            return True, params_dict
        except Exception as e:
            print(f"Model generation failed: {str(e)}")
            return False, None

    def calculate_lift(self):
        """计算升力"""
        try:
            with TimeoutGuard(OPERATION_TIMEOUT) as guard:
                grid_bounds = get_grid_aabb()
                if guard.timed_out:
                    raise TimeoutError("Get grid bounds timed out")
            
            x1, y1, z1, x2, y2, z2 = [convert_numpy_types(b) for b in grid_bounds]
            
            # 网格规模检查
            grid_size = (x2-x1) * (y2-y1) * (z2-z1)
            if grid_size > MAX_GRID_CELLS:
                raise ValueError(f"Grid too large ({grid_size} cells)")
            
            # 动态调整采样步长，增加最小步长约束
            sample_step = max(2, int(grid_size ** 0.33 / 40))  # 步长更大，减少计算波动
            total_lift = 0.0

            # 增加邻域平均（平滑升力计算）
            for x in range(x1, x2+1, sample_step):
                for y in range(y1, y2+1, sample_step):
                    for z in range(z1, z2+1, sample_step):
                        if cell(x, y, z) != 0:
                            # 3x3x3邻域采样
                            neighbor_lift = 0
                            count = 0
                            for dx in [-1,0,1]:
                                for dy in [-1,0,1]:
                                    for dz in [-1,0,1]:
                                        if cell(x+dx, y+dy, z+dz) != 0:
                                            aoa = self.estimate_aoa(x+dx, y+dy, z+dz)
                                            area_cell = (GRID_UNIT_SIZE / 100) **2  
                                            cl = self.lift_coefficient(aoa)
                                            neighbor_lift += 0.5 * AIR_DENSITY * (MAX_VELOCITY**2) * area_cell * cl 
                                            count +=1
                            if count >0:
                                total_lift += neighbor_lift / count * (sample_step**3)
            
            return max(total_lift, 0.1)  # 避免升力为0
        except Exception as e:
            print(f"Lift calculation failed: {str(e)}")
            return 0.0

    def estimate_aoa(self, x, y, z):
        """估算迎角"""
        neighbors = []
        for dx, dy, dz in [(1,0,0), (0,0,1)]:
            if cell(x + dx, y + dy, z + dz) == 0:
                neighbors.append((dx, dy, dz))
        
        if not neighbors:
            return math.radians(5)
        
        normal = [sum(d[0] for d in neighbors), 0, sum(d[2] for d in neighbors)]
        normal_mag = math.sqrt(normal[0]**2 + normal[2]**2)
        return min(math.acos(abs(normal[2])/normal_mag) if normal_mag else 0, math.radians(15))

    def lift_coefficient(self, aoa):
        """升力系数计算"""
        aoa_deg = math.degrees(aoa)
        # 更精确的升力系数模型
        if aoa_deg < 12:
            return 0.1 * aoa_deg  # 线性区域
        elif aoa_deg < 18:
            return 1.2 - 0.05 * (aoa_deg - 12) ** 2  # 非线性区域，开始失速
        else:
            return 0.8  # 完全失速

    def evaluate(self, params):
        """评估函数：以升力/面积比为优化目标"""
        self.iteration += 1
        
        # 打印当前参数
        params_dict = dict(zip(self.param_names, params))
        print(f"\nIteration {self.iteration}/{MAX_ITERATIONS} Parameters:")
        print(f"Root Chord: {params_dict['width1']:.2f}dm, Tip Chord: {params_dict['width2']:.2f}dm, Span: {params_dict['length']:.2f}dm")
        print(f"Root Angle: {params_dict['angle1']:.2f}°, Tip Angle: {params_dict['angle2']:.2f}°")
        
        try:
            # 生成模型并检查约束
            success, params_dict = self.safe_generate_model(params)
            if not success:
                self.failed_attempts += 1
                if self.failed_attempts >= self.max_failures:
                    print(f"Consecutive {self.max_failures} failures, narrowing parameter range")
                    self.dimensions = [
                        Real(name='width1', low=2.2, high=4.8),
                        Real(name='width2', low=0.7, high=2.3),
                        Real(name='length', low=16.0, high=28.0),
                        Real(name='angle1', low=6.0, high=18.0),
                        Real(name='angle2', low=0.0, high=12.0)
                    ]
                return 100.0  # 惩罚失败案例
            
            # 计算面积和升力
            wing_area = self.calculate_wing_area(params_dict)
            lift = self.calculate_lift()
            
            # 如果升力为0，视为无效样本
            if lift <= 0.1:
                print("Invalid sample (zero lift detected), skipping...")
                return 100.0
            
            lift = max(lift, 0.1)  # 避免升力为0
            
            # 计算升力/面积比（核心优化目标）
            lift_area_ratio = lift / wing_area
            
            # 记录历史
            self.history.append({
                'iteration': self.iteration,
                'params': {k: round(v, 2) for k, v in params_dict.items()},
                'lift': round(lift, 2),
                'area': round(wing_area, 4),
                'ratio': round(lift_area_ratio, 2),
                'time': datetime.now().strftime("%H:%M:%S")
            })
            
            # 只记录有效迭代数据
            self.valid_iterations.append(self.iteration)
            self.valid_ratios.append(lift_area_ratio)
            
            # 更新最优解
            if lift_area_ratio > self.best_ratio + 0.05:  # 增加最小改进阈值
                self.best_ratio = lift_area_ratio
                self.best_params = params_dict.copy()
                self.best_lift = lift
                self.best_area = wing_area
                self.last_successful = params_dict.copy()
                print(f"New best efficiency: Lift={lift:.2f}N, Area={wing_area:.4f}m2, Ratio={lift_area_ratio:.2f}N/m2")
                self.stall_count = 0  # 重置停滞计数
            else:
                self.stall_count += 1
                print(f"Current efficiency: Lift={lift:.2f}N, Area={wing_area:.4f}m2, Ratio={lift_area_ratio:.2f}N/m2")
                print(f"Best ratio stalled {self.stall_count}/{self.max_stall} times")
            
            # 更新最佳比值历史
            if self.best_ratio_history:
                self.best_ratio_history.append(max(self.best_ratio_history[-1], lift_area_ratio))
            else:
                self.best_ratio_history.append(lift_area_ratio)
            
            self.failed_attempts = 0
            # 返回负的比值（因为gp_minimize是最小化函数）
            return -lift_area_ratio
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            self.failed_attempts += 1
            return 100.0  # 惩罚错误案例

    def optimize(self):
        """执行优化"""
        print("Starting wing efficiency optimization (maximizing lift/area ratio)...")
        
        @use_named_args(dimensions=self.dimensions)
        def objective(**params):
            return self.evaluate([params[name] for name in self.param_names])
        
        try:
            # 使用贝叶斯优化（基于高斯过程的GP优化）
            result = gp_minimize(
                func=objective,
                dimensions=self.dimensions,
                n_calls=MAX_ITERATIONS,
                random_state=42,
                acq_func='EI',  # 期望改进
                acq_optimizer='lbfgs',  # 使用L-BFGS优化采集函数
                n_initial_points=10,  # 增加初始点数量
                noise=0.05,  # 减小噪声
                callback=self.print_progress,
                n_jobs=1  # 串行执行
            )
            
            if result.x:
                self.best_params = convert_numpy_types(dict(zip(self.param_names, result.x)))
                self.best_ratio = -result.fun
            self.save_results()
            self.plot_optimization_curve()  # 绘制优化曲线
        except Exception as e:
            print(f"Optimization interrupted: {str(e)}")
            self.save_results()
            self.plot_optimization_curve()  # 绘制已完成的优化曲线

    def print_progress(self, res):
        current_best = max(self.best_ratio, -res.fun)
        print(f"Progress: Iteration {self.iteration}/{MAX_ITERATIONS}, Best Ratio: {current_best:.2f}N/m2")

    def save_results(self):
        """保存结果"""
        if not self.best_params and self.last_successful:
            self.best_params = self.last_successful
            print("Using last successful parameters as result")
        
        if self.best_params:
            print("\n=== Optimization Results ===")
            print(f"Best Lift/Area Ratio: {self.best_ratio:.2f} N/m2")
            print(f"Corresponding Lift: {self.best_lift:.2f} N, Wing Area: {self.best_area:.4f} m2")
            print("Optimal Parameters:")
            for k, v in self.best_params.items():
                print(f"  {k:8}: {v:.2f}")
            
            # 保存最优模型
            if self.safe_generate_model(list(self.best_params.values()), "optimal_efficient_wing")[0]:
                print("Optimal model saved: optimal_efficient_wing.sm")
        
        # 保存历史
        history_data = convert_numpy_types({
            'metadata': {
                'best_ratio': self.best_ratio,
                'best_lift': self.best_lift,
                'best_area': self.best_area,
                'iterations': self.iteration
            },
            'history': self.history
        })
        
        with open("efficiency_optimization_history.json", "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)
        print("Optimization history saved")

    def plot_optimization_curve(self):
        """绘制优化曲线（只显示有效样本点）"""
        if not self.valid_iterations:
            print("No valid data points to plot optimization curve")
            return
        
        plt.figure(figsize=(12, 6))
        
        # 绘制当前迭代的升力/面积比（只显示有效点）
        plt.plot(self.valid_iterations, self.valid_ratios, 'bo-', alpha=0.6, label='Valid Samples')
        
        # 绘制最优升力/面积比的变化
        plt.plot(self.valid_iterations, self.best_ratio_history, 'r-', linewidth=2, label='Best Ratio')
        
        plt.title('Wing Optimization Process: Lift/Area Ratio vs Iterations', fontsize=14)
        plt.xlabel('Iteration Count', fontsize=12)
        plt.ylabel('Lift/Area Ratio (N/m2)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # 确保x轴为整数
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig('wing_optimization_curve.png', dpi=300)
        print("Optimization curve saved: wing_optimization_curve.png")
        plt.show()

def main():
    print("=== Wing Efficiency Optimization Program ===")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化环境
    DoPHG(f"""
        sets('mesh_path', 'C:/Users/18858/Desktop/');
        sets('grid_path', 'C:/Users/18858/Desktop/');
        setf('unit_size', {GRID_UNIT_SIZE});
        echo(0);
    """)
    
    optimizer = WingEfficiencyOptimizer()
    optimizer.optimize()

if __name__ == "__main__":
    main()