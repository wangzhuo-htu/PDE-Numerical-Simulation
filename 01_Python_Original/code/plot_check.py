# plot_check.py
# 根据 check1D.py 跑出来的最新数据生成图17（数值收敛）和图18（对比与消融分析）
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

pal = ['#2E5A88', '#16A085', '#E67E22', '#C0392B', '#8E44AD']

def style_ax(ax):
    ax.tick_params(direction='in', which='both', top=True, right=True, width=1.2, labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    for s in ax.spines.values(): s.set_linewidth(1.2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

# ================= 填入 check1D.py 跑出来的最新数据 =================
# 1. 网格无关性
N_vals = [50, 100, 200]
t_N = [56.17, 56.65, 56.90]

# 2. 时间步长收敛性
dt_vals = [120, 60, 30]
t_dt = [56.67, 56.65, 56.63]

# 3. 灵敏度分析
h_vals = [22.5, 25.0, 27.5]
T_surface = [35.1007, 35.4941, 35.8470]

# 4. 对比与消融
labels = ['全域最大\n(56.65h)', '平均含水率\n(35.10h)', '固定半径\n(56.65h)', '动态收缩\n(22.52h)']
times = [56.65, 35.10, 56.65, 22.52]
colors = ['#2E5A88', '#E67E22', '#7F8C8D', '#C0392B']

# ================= 图17：数值收敛性检验（2张子图） =================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axes[0]
ax.plot(N_vals, t_N, 'o-', color=pal[0], linewidth=2.2, markersize=8)
ax.set_xlabel('网格数 N'); ax.set_ylabel('烘干时间 / h')
ax.set_title('(a) 网格无关性检验', fontsize=12, fontweight='bold')
ax.set_xticks(N_vals)
for i, txt in enumerate(t_N):
    ax.annotate(f'{txt:.2f}h', (N_vals[i], t_N[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
style_ax(ax)

ax = axes[1]
ax.plot(dt_vals, t_dt, 's-', color=pal[1], linewidth=2.2, markersize=8)
ax.set_xlabel('时间步长 dt / s'); ax.set_ylabel('烘干时间 / h')
ax.set_title('(b) 时间步长收敛性检验', fontsize=12, fontweight='bold')
ax.set_xticks(dt_vals); ax.set_xticklabels([str(d) for d in dt_vals])
for i, txt in enumerate(t_dt):
    ax.annotate(f'{txt:.2f}h', (dt_vals[i], t_dt[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
style_ax(ax)

plt.tight_layout()
plt.savefig('图17_模型检验_数值收敛性.png', dpi=300, bbox_inches='tight')
plt.close()
print('图17 已保存：图17_模型检验_数值收敛性.png')

# ================= 图18：对比与消融分析（2张子图） =================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axes[0]
ax.plot(h_vals, T_surface, '^-', color=pal[3], linewidth=2.2, markersize=8)
ax.set_xlabel('对流换热系数 h / (W/(m²·K))')
ax.set_ylabel('1800s 表面温度 / ℃')
ax.set_title('(a) 灵敏度分析', fontsize=12, fontweight='bold')
ax.set_xticks(h_vals)
for i, txt in enumerate(T_surface):
    ax.annotate(f'{txt:.2f}℃', (h_vals[i], T_surface[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
style_ax(ax)

ax = axes[1]
bars = ax.bar(labels, times, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
for bb, tt in zip(bars, times):
    ax.text(bb.get_x() + bb.get_width()/2, bb.get_height() + 1.0, f'{tt:.2f} h',
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('烘干时间 / h')
ax.set_title('(b) 终点判据对比与尺寸收缩消融', fontsize=12, fontweight='bold')
ax.set_ylim(0, 68)
style_ax(ax)

plt.tight_layout()
plt.savefig('图18_模型检验_对比与消融.png', dpi=300, bbox_inches='tight')
plt.close()
print('图18 已保存：图18_模型检验_对比与消融.png')
print('模型检验绘图完成！')