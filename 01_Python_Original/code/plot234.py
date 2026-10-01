# plot234.py
# 统一绘制问题2、3、4的全部图表（图7~图16）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ====== 全局字体与风格（防止方框、乱码） ======
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# 统一色板
pal = ['#2E5A88', '#16A085', '#E67E22', '#C0392B', '#8E44AD', '#7D3C98']

def style_ax(ax):
    """统一坐标轴风格，防数字重叠"""
    ax.tick_params(direction='in', which='both', top=True, right=True, width=1.2, labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    for s in ax.spines.values(): s.set_linewidth(1.2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

# ====== 读取数据 ======
dfT2 = pd.read_excel('result2.xlsx', sheet_name='温度')
dfC2 = pd.read_excel('result2.xlsx', sheet_name='水分浓度')
dfC3 = pd.read_excel('result3.xlsx', sheet_name='水分浓度')
dfC4 = pd.read_excel('result4.xlsx', sheet_name='水分浓度')

t2 = dfT2['时间'].values / 3600
t3 = dfC3['时间'].values / 3600
t4 = dfC4['时间'].values / 3600

T2 = dfT2.iloc[:, 1:].values
C2 = dfC2.iloc[:, 1:].values
C3 = dfC3.iloc[:, 1:].values
C4 = dfC4.iloc[:, 1:-1].values      # 距离列
C4_surf = dfC4.iloc[:, -1].values   # 表面列

# 半径数据（问题四用）
rad = pd.read_excel('附件2.xlsx')
fR = interp1d(rad['时间'], rad['半径'] / 100.0, kind='linear', fill_value='extrapolate')
R4 = fR(dfC4['时间'].values)

rOut = np.arange(0, 2.01, 0.1)
tShow = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# =====================================================
# 图7：问题2温度径向分布
# =====================================================
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for i, hh in enumerate(tShow):
    idx = int(hh * 3600)
    ax.plot(rOut, T2[idx, :], marker='o', markersize=4, linewidth=1.8,
            color=pal[i % 6], markeredgecolor='white', markeredgewidth=0.6,
            label=f't={hh}h')
ax.set_xlabel('到药材中心距离 / cm'); ax.set_ylabel('温度 / ℃')
ax.set_title('图7  问题2温度沿半径分布', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray', fontsize=9, ncol=2)
style_ax(ax); plt.tight_layout()
plt.savefig('图7_问题2温度径向.png', dpi=300, bbox_inches='tight'); plt.close()

# =====================================================
# 图8：问题2水分径向分布
# =====================================================
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for i, hh in enumerate(tShow):
    idx = int(hh * 3600)
    ax.plot(rOut, C2[idx, :], marker='s', markersize=4, linewidth=1.8,
            color=pal[i % 6], markeredgecolor='white', markeredgewidth=0.6,
            label=f't={hh}h')
ax.set_xlabel('到药材中心距离 / cm'); ax.set_ylabel('水分浓度 / (kg/kg)')
ax.set_title('图8  问题2水分浓度沿半径分布', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=9, ncol=2)
style_ax(ax); plt.tight_layout()
plt.savefig('图8_问题2水分径向.png', dpi=300, bbox_inches='tight'); plt.close()

# =====================================================
# 图9：问题2三维曲面（防重叠）
# =====================================================
fig = plt.figure(figsize=(14, 5.5))
ax1 = fig.add_subplot(121, projection='3d')
X, Y = np.meshgrid(rOut, t2)
surf1 = ax1.plot_surface(X, Y, T2, cmap='plasma', rstride=1, cstride=1, antialiased=True)
ax1.set_xlabel('半径 / cm', labelpad=8); ax1.set_ylabel('时间 / h', labelpad=8); ax1.set_zlabel('温度 / ℃', labelpad=8)
ax1.set_title('(a) 温度三维分布', fontsize=12, fontweight='bold')
ax1.view_init(elev=25, azim=-60); ax1.tick_params(labelsize=9)
ax1.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); ax1.set_yticks([0, 1, 2, 3])
fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1, label='温度 / ℃')

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, C2, cmap='turbo', rstride=1, cstride=1, antialiased=True)
ax2.set_xlabel('半径 / cm', labelpad=8); ax2.set_ylabel('时间 / h', labelpad=8); ax2.set_zlabel('水分浓度 / (kg/kg)', labelpad=8)
ax2.set_title('(b) 水分浓度三维分布', fontsize=12, fontweight='bold')
ax2.view_init(elev=25, azim=-60); ax2.tick_params(labelsize=9)
ax2.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); ax2.set_yticks([0, 1, 2, 3])
fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1, label='水分浓度 / (kg/kg)')

plt.tight_layout()
plt.savefig('图9_问题2三维曲面.png', dpi=300, bbox_inches='tight'); plt.close()

# =====================================================
# 图10：问题2热力图（防乱码）
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
im1 = axes[0].contourf(t2, rOut, T2.T, levels=40, cmap='plasma')
axes[0].set_xlabel('时间 / h'); axes[0].set_ylabel('到药材中心距离 / cm')
axes[0].set_title('(a) 温度时空分布', fontsize=12, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=axes[0], label='温度 / ℃'); cbar1.ax.tick_params(labelsize=9)
im2 = axes[1].contourf(t2, rOut, C2.T, levels=40, cmap='turbo')
axes[1].set_xlabel('时间 / h'); axes[1].set_ylabel('到药材中心距离 / cm')
axes[1].set_title('(b) 水分浓度时空分布', fontsize=12, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=axes[1], label='水分浓度 / (kg/kg)'); cbar2.ax.tick_params(labelsize=9)
for ax in axes: style_ax(ax)
plt.tight_layout()
plt.savefig('图10_问题2热力图.png', dpi=300, bbox_inches='tight'); plt.close()

# =====================================================
# ==================== 图11：问题3水分时间演化 ====================
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(t3, C3[:, 0], linewidth=2.2, color='#1f4e79', label='中心 r=0cm')
ax.plot(t3, C3[:, -1], linewidth=2.2, color='#c00000', label='表面 r=2cm')
ax.axhline(0.15, color='#7f7f7f', linestyle='--', linewidth=1.5, label='干燥阈值 0.15')
ax.set_xlabel('时间 / h')
ax.set_ylabel('水分浓度 / (kg/kg)')
#ax.set_title('图11  问题3不同位置水分浓度随时间变化', fontsize=13, fontweight='bold')
ax.set_ylim(0.04, 2.6)   # 让表面限幅段贴底，不显突兀
ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=10)
style_ax(ax)
plt.tight_layout()
plt.savefig('图11_问题3水分时间.png', dpi=300, bbox_inches='tight')
plt.close()

# =====================================================
# 图12：问题3热力图（防乱码、高对比度）
# =====================================================
fig, ax = plt.subplots(figsize=(9, 5.2))
im = ax.contourf(t3, rOut, C3.T, levels=40, cmap='coolwarm')
contours = ax.contour(t3, rOut, C3.T, levels=6, colors='white', linewidths=0.6, alpha=0.8)
ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f', colors='black')
ax.set_xlabel('时间 / h'); ax.set_ylabel('到药材中心距离 / cm')
ax.set_title('图12  问题3水分浓度时空分布', fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cbar.set_label('水分浓度 / (kg/kg)', fontsize=11); cbar.ax.tick_params(labelsize=10)
style_ax(ax); plt.tight_layout()
plt.savefig('图12_问题3热力图.png', dpi=300, bbox_inches='tight'); plt.close()

# =====================================================
# 图13：问题3结束时水分分布
# =====================================================
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.bar(rOut, C3[-1, :], width=0.08, color='#1f4e79', edgecolor='black', linewidth=0.6)
ax.axhline(0.15, color='red', linestyle='--', linewidth=1.5, label='阈值 0.15')
ax.set_xlabel('到药材中心距离 / cm'); ax.set_ylabel('水分浓度 / (kg/kg)')
ax.set_title(f'图13  问题3烘干结束时（{t3[-1]:.2f}h）水分分布', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')
style_ax(ax); plt.tight_layout()
plt.savefig('图13_问题3结束分布.png', dpi=300, bbox_inches='tight'); plt.close()

# =====================================================
# 图14：问题4消融对比
# =====================================================
fig, ax = plt.subplots(figsize=(6.5, 4.8))
models = ['对照模型\n(固定2cm)', '完整模型\n(动态收缩)']
times = [56.63, 22.52]
bars = ax.bar(models, times, color=['#7F8C8D', '#2E5A88'], edgecolor='black', linewidth=0.8, width=0.5)
for bb, tt in zip(bars, times):
    ax.text(bb.get_x() + bb.get_width()/2, bb.get_height() + 1.5, f'{tt:.2f} h',
            ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('烘干时间 / h')
ax.set_title('图14  尺寸收缩对烘干时间的消融对比', fontsize=13, fontweight='bold')
ax.set_ylim(0, 65)
style_ax(ax); plt.tight_layout()
plt.savefig('图14_问题4消融对比.png', dpi=300, bbox_inches='tight'); plt.close()

# =====================================================
# 图15：问题4水分随时间变化
# =====================================================
fig, ax = plt.subplots(figsize=(8, 4.8))
rShow = [0, 0.5, 1.0]
for i, rr in enumerate(rShow):
    col = int(rr / 0.1)
    ax.plot(t4, C4[:, col], linewidth=2.2, color=['#1f4e79','#16A085','#E67E22'][i], label=f'r={rr}cm')
ax.plot(t4, C4_surf, linewidth=2.2, color='#c00000', linestyle='--', label='药材表面')
ax.axhline(0.15, color='#7f7f7f', linestyle=':', linewidth=1.5, label='干燥阈值 0.15')
ax.set_xlabel('时间 / h'); ax.set_ylabel('水分浓度 / (kg/kg)')
ax.set_title('图15  问题4收缩条件下不同位置水分随时间变化', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=9, ncol=2)
style_ax(ax); plt.tight_layout()
plt.savefig('图15_问题4水分时间.png', dpi=300, bbox_inches='tight'); plt.close()

# =====================================================
# 图16：问题4半径与水分耦合
# =====================================================
fig, ax1 = plt.subplots(figsize=(8, 4.8))
cR = '#c00000'; cC = '#1f4e79'
ax1.set_xlabel('时间 / h')
ax1.set_ylabel('药材半径 / cm', color=cR)
ax1.plot(t4, R4 * 100, color=cR, linewidth=2.2, label='半径 R(t)')
ax1.tick_params(axis='y', labelcolor=cR, direction='in', width=1.2)
ax2 = ax1.twinx()
ax2.set_ylabel('全域最大水分浓度 / (kg/kg)', color=cC)
maxC = np.max(C4, axis=1)
ax2.plot(t4, maxC, color=cC, linewidth=2.2, linestyle='--', label='全域最大水分浓度')
ax2.axhline(0.15, color='#7f7f7f', linestyle=':', linewidth=1.5)
ax2.tick_params(axis='y', labelcolor=cC, direction='in', width=1.2)
ax1.set_title('图16  半径收缩与水分浓度下降的耦合演化', fontsize=13, fontweight='bold')
l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc='center right', framealpha=0.9, edgecolor='none', fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.4, color='gray')
for s in ax1.spines.values(): s.set_linewidth(1.2)
for s in ax2.spines.values(): s.set_linewidth(1.2)
plt.tight_layout()
plt.savefig('图16_问题4半径水分耦合.png', dpi=300, bbox_inches='tight'); plt.close()

print('plot234 全部完成：图7 ~ 图16 已生成')