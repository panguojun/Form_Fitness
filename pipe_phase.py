import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.interpolate import interp1d

class PhaseDiagramGenerator:
    def __init__(self):
        # 初始化图形
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.3)
        
        # 初始参数
        self.initial_temp = 300  # 初始温度(K)
        self.initial_pressure = 1  # 初始压力(atm)
        self.external_field = 0  # 外部场(如磁场、电场等)
        
        # 创建滑块
        ax_temp = plt.axes([0.2, 0.2, 0.6, 0.03])
        ax_pressure = plt.axes([0.2, 0.15, 0.6, 0.03])
        ax_field = plt.axes([0.2, 0.1, 0.6, 0.03])
        
        self.slider_temp = Slider(ax_temp, 'Temperature (K)', 100, 1000, valinit=self.initial_temp)
        self.slider_pressure = Slider(ax_pressure, 'Pressure (atm)', 0.1, 100, valinit=self.initial_pressure)
        self.slider_field = Slider(ax_field, 'External Field', -10, 10, valinit=self.external_field)
        
        # 添加重置按钮
        reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
        
        # 绑定事件
        self.slider_temp.on_changed(self.update)
        self.slider_pressure.on_changed(self.update)
        self.slider_field.on_changed(self.update)
        self.reset_button.on_clicked(self.reset)
        
        # 初始绘图
        self.update(None)
        
    def calculate_phase_boundaries(self, temp, pressure, field):
        """计算不同条件下的相边界"""
        # 这里使用简化的模型，实际应用中应替换为真实的物理模型
        
        # 固态-液态边界 (简化Clausius-Clapeyron方程)
        t_melt = np.linspace(200, 1000, 100)
        p_melt = 0.1 * np.exp(0.01*(t_melt-273)) + 0.5*field
        
        # 液态-气态边界
        t_vapor = np.linspace(300, 647, 100)  # 647K是水的临界温度
        p_vapor = 1.0 * np.exp(-5000*(1/t_vapor - 1/647)) + 0.3*field
        
        # 固态-气态边界 (升华曲线)
        t_sublime = np.linspace(100, 273, 50)
        p_sublime = 0.01 * np.exp(0.02*(t_sublime-100)) + 0.2*field
        
        # 根据当前条件调整边界
        if temp > 500 and pressure > 50:
            # 高温高压下可能出现新相
            t_new_phase = np.linspace(500, 800, 50)
            p_new_phase = 50 + 0.5*(t_new_phase-500) + 0.8*field
            new_phase_exists = True
        else:
            new_phase_exists = False
            
        return {
            'melt': (t_melt, p_melt),
            'vapor': (t_vapor, p_vapor),
            'sublime': (t_sublime, p_sublime),
            'new_phase': (t_new_phase, p_new_phase) if new_phase_exists else None,
            'new_phase_exists': new_phase_exists
        }
    
    def update(self, val):
        """更新图表"""
        temp = self.slider_temp.val
        pressure = self.slider_pressure.val
        field = self.slider_field.val
        
        # 清除当前图形
        self.ax.clear()
        
        # 计算相边界
        boundaries = self.calculate_phase_boundaries(temp, pressure, field)
        
        # 绘制相边界
        self.ax.plot(*boundaries['melt'], 'b-', label='Solid-Liquid')
        self.ax.plot(*boundaries['vapor'], 'r-', label='Liquid-Gas')
        self.ax.plot(*boundaries['sublime'], 'g-', label='Solid-Gas')
        
        if boundaries['new_phase_exists']:
            self.ax.plot(*boundaries['new_phase'], 'm--', label='New Phase')
        
        # 标记当前状态点
        self.ax.plot(temp, pressure, 'ko', markersize=10)
        self.ax.annotate(f'Current State\nT={temp:.1f} K\nP={pressure:.1f} atm', 
                         (temp, pressure), textcoords="offset points", xytext=(10,10), 
                         ha='left', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        # 设置图形属性
        self.ax.set_xlabel('Temperature (K)')
        self.ax.set_ylabel('Pressure (atm)')
        self.ax.set_title('Interactive Phase Diagram')
        self.ax.set_xlim(100, 1000)
        self.ax.set_ylim(0.01, 100)
        self.ax.set_yscale('log')
        self.ax.grid(True, which="both", ls="-", alpha=0.5)
        self.ax.legend(loc='upper left')
        
        # 根据当前状态标注相区
        if boundaries['new_phase_exists'] and temp > 500 and pressure > 50:
            self.ax.text(700, 70, 'New Phase Region', fontsize=12, bbox=dict(facecolor='purple', alpha=0.2))
        elif pressure < boundaries['sublime'][1][-1] and temp < 273:
            self.ax.text(150, 0.1, 'Solid Region', fontsize=12, bbox=dict(facecolor='blue', alpha=0.2))
        elif pressure > boundaries['vapor'][1][0] and temp < 647:
            self.ax.text(400, 10, 'Liquid Region', fontsize=12, bbox=dict(facecolor='red', alpha=0.2))
        else:
            self.ax.text(600, 0.1, 'Gas Region', fontsize=12, bbox=dict(facecolor='green', alpha=0.2))
        
        self.fig.canvas.draw_idle()
    
    def reset(self, event):
        """重置滑块到初始值"""
        self.slider_temp.reset()
        self.slider_pressure.reset()
        self.slider_field.reset()

# 运行相图生成器
if __name__ == "__main__":
    pdg = PhaseDiagramGenerator()
    plt.show()