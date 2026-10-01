# 问题一绘图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

dfT = pd.read_excel('result1.xlsx', sheet_name='温度')
dfC = pd.read_excel('result1.xlsx', sheet_name='水分浓度')
tAll = dfT['时间'].values
To = dfT.iloc[:, 1:].values
Co = dfC.iloc[:, 1:].values
rOut = np.arange(0, 2.01, 0.1)

air = pd.read_excel('附件1.xlsx')
fTa = interp1d(air['时间'], air['温度'], kind='linear', fill_value='extrapolate')
fCa = interp1d(air['时间'], air['水分浓度'], kind='linear', fill_value='extrapolate')
tAir = np.arange(0, 1801, 1)
TaAir = fTa(tAir); CaAir = fCa(tAir)

tShow = [100, 300, 600, 900, 1200, 1500, 1800]
rShow = [0, 0.5, 1, 1.5, 2]
pal = ['#2E5A88', '#16A085', '#E67E22', '#C0392B', '#8E44AD']

def style_ax(ax):
    ax.tick_params(direction='in', which='both', top=True, right=True, width=1.2, labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    for s in ax.spines.values(): s.set_linewidth(1.2)

# 图1 烘房条件
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(tAir, TaAir, color='#C0392B', linewidth=2)
axes[0].set_xlabel('时间/s'); axes[0].set_ylabel('温度/℃')
axes[0].set_title('(a) 烘房空气温度', fontsize=12, fontweight='bold')
axes[1].plot(tAir, CaAir, color='#2E5A88', linewidth=2)
axes[1].set_xlabel('时间/s'); axes[1].set_ylabel('水分浓度/(kg/kg)')
axes[1].set_title('(b) 烘房空气水分浓度', fontsize=12, fontweight='bold')
for ax in axes: style_ax(ax)
fig.suptitle('图1  烘房空气条件随时间变化', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('图1_烘房边界条件.png', dpi=300, bbox_inches='tight'); plt.close()

# 图2 温度径向
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for i, tt in enumerate(tShow):
    ax.plot(rOut, To[tt, :], marker='o', markersize=5, linewidth=2, color=pal[i % 5],
            markeredgecolor='white', markeredgewidth=0.8, label=f't={tt}s')
ax.set_xlabel('到药材中心距离/cm'); ax.set_ylabel('温度/℃')
ax.set_title('图2  不同时刻温度沿半径分布', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray', fontsize=9)
style_ax(ax); plt.tight_layout(); plt.savefig('图2_温度径向分布.png', dpi=300, bbox_inches='tight'); plt.close()

# 图3 温度时间
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for i, rr in enumerate(rShow):
    ax.plot(tAll, To[:, int(rr/0.1)], linewidth=2, color=pal[i % 5],
            markeredgecolor='white', markeredgewidth=0.8, label=f'r={rr}cm')
ax.set_xlabel('时间/s'); ax.set_ylabel('温度/℃')
ax.set_title('图3  不同位置温度随时间变化', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9, ncol=2, edgecolor='gray', fontsize=9)
style_ax(ax); plt.tight_layout(); plt.savefig('图3_温度时间变化.png', dpi=300, bbox_inches='tight'); plt.close()

# 图4 水分径向
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for i, tt in enumerate(tShow):
    ax.plot(rOut, Co[tt, :], marker='s', markersize=5, linewidth=2, color=pal[i % 5],
            markeredgecolor='white', markeredgewidth=0.8, label=f't={tt}s')
ax.set_xlabel('到药材中心距离/cm'); ax.set_ylabel('水分浓度/(kg/kg)')
ax.set_title('图4  不同时刻水分浓度沿半径分布', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=9)
style_ax(ax); plt.tight_layout(); plt.savefig('图4_水分径向分布.png', dpi=300, bbox_inches='tight'); plt.close()

# 图5 水分时间
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for i, rr in enumerate(rShow):
    ax.plot(tAll, Co[:, int(rr/0.1)], linewidth=2, color=pal[i % 5],
            markeredgecolor='white', markeredgewidth=0.8, label=f'r={rr}cm')
ax.set_xlabel('时间/s'); ax.set_ylabel('水分浓度/(kg/kg)')
ax.set_title('图5  不同位置水分浓度随时间变化', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9, ncol=2, edgecolor='gray', fontsize=9)
style_ax(ax); plt.tight_layout(); plt.savefig('图5_水分时间变化.png', dpi=300, bbox_inches='tight'); plt.close()

# 图6 + 图7 横截面云图
theta = np.linspace(0, 2 * np.pi, 200)
rg = np.linspace(0, 2, 21)
TH, RG = np.meshgrid(theta, rg)
XX = RG * np.cos(TH); YY = RG * np.sin(TH)
tSnap = [100, 600, 1200, 1800]

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
tMin = To[tSnap[0], :].min(); tMax = To[tSnap[-1], :].max()
levT = np.linspace(tMin, tMax, 40)
for ax, tt in zip(axes.flatten(), tSnap):
    ZZ = np.tile(To[tt, :].reshape(-1, 1), (1, len(theta)))
    cs = ax.contourf(XX, YY, ZZ, levels=levT, cmap='plasma', extend='both')
    ax.contour(XX, YY, ZZ, levels=levT[::4], colors='white', linewidths=0.5, alpha=0.6)
    ax.add_patch(plt.Circle((0, 0), 2, fill=False, edgecolor='k', linewidth=1.5))
    ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)
    ax.set_aspect('equal'); ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3)
    ax.set_xlabel('x/cm', fontsize=10); ax.set_ylabel('y/cm', fontsize=10)
    ax.set_title(f't = {tt}s', fontsize=12, fontweight='bold')
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, label='温度/℃')
fig.suptitle('图6  不同时刻药材横截面温度分布', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(); plt.savefig('图6_圆形截面温度.png', dpi=300, bbox_inches='tight'); plt.close()

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
cMin = Co[tSnap[-1], :].min(); cMax = Co[tSnap[0], :].max()
levC = np.linspace(cMin, cMax, 40)
for ax, tt in zip(axes.flatten(), tSnap):
    ZZ = np.tile(Co[tt, :].reshape(-1, 1), (1, len(theta)))
    cs = ax.contourf(XX, YY, ZZ, levels=levC, cmap='viridis', extend='both')
    ax.contour(XX, YY, ZZ, levels=levC[::4], colors='white', linewidths=0.5, alpha=0.6)
    ax.add_patch(plt.Circle((0, 0), 2, fill=False, edgecolor='k', linewidth=1.5))
    ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)
    ax.set_aspect('equal'); ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3)
    ax.set_xlabel('x/cm', fontsize=10); ax.set_ylabel('y/cm', fontsize=10)
    ax.set_title(f't = {tt}s', fontsize=12, fontweight='bold')
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, label='水分浓度/(kg/kg)')
fig.suptitle('图7  不同时刻药材横截面水分浓度分布', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(); plt.savefig('图7_圆形截面水分.png', dpi=300, bbox_inches='tight'); plt.close()

print('plot1 全部完成')

#
# # 图8：三维曲面（正交投影 + 正方体比例 + 网格增强立体感）
# fig = plt.figure(figsize=(14, 6))
#
# # (a) 温度三维
# ax1 = fig.add_subplot(121, projection='3d')
# sk = 20
# tG, rG = np.meshgrid(tAll[::sk] / 3600, rOut)
# sf1 = ax1.plot_surface(tG, rG, To[::sk, :].T, cmap='plasma', edgecolor='none', alpha=0.95, antialiased=True)
#
# ax1.set_xlabel('时间/h', labelpad=10, fontsize=11)
# ax1.set_ylabel('半径/cm', labelpad=10, fontsize=11)
# ax1.set_zlabel('温度/℃', labelpad=10, fontsize=11)
# ax1.set_title('(a) 温度三维分布', fontsize=13, fontweight='bold')
#
# # 关键1：启用正交投影，彻底消除“歪斜”感，让坐标轴垂直
# ax1.set_proj_type('ortho')
# # 关键2：设置方正的盒体比例（X轴稍微长一点，Y和Z保持等比例）
# ax1.set_box_aspect((1.3, 1, 1))
#
# ax1.view_init(elev=30, azim=-50)  # 经典的俯视视角
#
# # 关键3：恢复网格线，但改成浅灰色虚线，保持立体感且不杂乱
# ax1.grid(True, linestyle=':', color='gray', alpha=0.4)
# # 背景板设为浅白/淡灰色，让正方体边界清晰可见
# ax1.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.1))
# ax1.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.1))
# ax1.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.1))
# ax1.xaxis.pane.set_edgecolor('gray')
# ax1.yaxis.pane.set_edgecolor('gray')
# ax1.zaxis.pane.set_edgecolor('gray')
#
# fig.colorbar(sf1, ax=ax1, shrink=0.6, label='温度/℃', pad=0.1)
#
# # (b) 水分三维
# ax2 = fig.add_subplot(122, projection='3d')
# sf2 = ax2.plot_surface(tG, rG, Co[::sk, :].T, cmap='viridis', edgecolor='none', alpha=0.95, antialiased=True)
#
# ax2.set_xlabel('时间/h', labelpad=10, fontsize=11)
# ax2.set_ylabel('半径/cm', labelpad=10, fontsize=11)
# ax2.set_zlabel('水分浓度/(kg/kg)', labelpad=10, fontsize=11)
# ax2.set_title('(b) 水分浓度三维分布', fontsize=13, fontweight='bold')
#
# # 同样启用正交投影和方正比例
# ax2.set_proj_type('ortho')
# ax2.set_box_aspect((1.3, 1, 1))
# ax2.view_init(elev=30, azim=-50)
#
# ax2.grid(True, linestyle=':', color='gray', alpha=0.4)
# ax2.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.1))
# ax2.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.1))
# ax2.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.1))
# ax2.xaxis.pane.set_edgecolor('gray')
# ax2.yaxis.pane.set_edgecolor('gray')
# ax2.zaxis.pane.set_edgecolor('gray')
#
# fig.colorbar(sf2, ax=ax2, shrink=0.6, label='水分浓度/(kg/kg)', pad=0.1)
#
# fig.suptitle('图8  药材温度与水分浓度的三维分布', fontsize=15, fontweight='bold')
# plt.tight_layout()
# plt.savefig('图8_三维曲面.png', dpi=300, bbox_inches='tight')
# plt.close()
# # 图9：时空热力图（升级版）
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # (a) 温度热力图
# im1 = axes[0].imshow(To.T, aspect='auto', origin='lower',
#                      extent=[0, 1800, 0, 2], cmap='plasma', interpolation='bilinear')
# axes[0].set_xlabel('时间/s', fontsize=11)
# axes[0].set_ylabel('到药材中心距离/cm', fontsize=11)
# axes[0].set_title('(a) 温度时空分布', fontsize=13, fontweight='bold')
# cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
# cbar1.set_label('温度/℃', fontsize=11)
# ct1 = axes[0].contour(tAll[::60], rOut, To[::60, :].T, levels=8,
#                       colors='white', linewidths=0.7, alpha=0.8)
# axes[0].clabel(ct1, inline=True, fontsize=8, fmt='%.1f')
#
# # (b) 水分热力图
# im2 = axes[1].imshow(Co.T, aspect='auto', origin='lower',
#                      extent=[0, 1800, 0, 2], cmap='viridis', interpolation='bilinear')
# axes[1].set_xlabel('时间/s', fontsize=11)
# axes[1].set_ylabel('到药材中心距离/cm', fontsize=11)
# axes[1].set_title('(b) 水分浓度时空分布', fontsize=13, fontweight='bold')
# cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
# cbar2.set_label('水分浓度/(kg/kg)', fontsize=11)
# ct2 = axes[1].contour(tAll[::60], rOut, Co[::60, :].T, levels=8,
#                       colors='white', linewidths=0.7, alpha=0.8)
# axes[1].clabel(ct2, inline=True, fontsize=8, fmt='%.3f')
#
# fig.suptitle('图9  药材温度与水分浓度的时空分布热力图', fontsize=15, fontweight='bold')
# plt.tight_layout()
# plt.savefig('图9_时空热力图.png', dpi=300, bbox_inches='tight')
# plt.close()