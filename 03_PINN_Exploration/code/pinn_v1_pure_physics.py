# -*- coding: utf-8 -*-
"""
基于 PINN 的圆柱形药材预热阶段一维径向热传导探索代码
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==============================
# 1. 基础设置
# ==============================

# 固定随机种子
torch.manual_seed(1234)
np.random.seed(1234)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 绘图设置，防止中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================
# 2. 物理参数提取
# ==============================
R = 0.02  # 半径 m
T0 = 28.0  # 初始温度 ℃
T_air_start = 28.0  # 初始空气温度 ℃
T_air_end = 41.513  # 1800s空气温度 ℃
t_end = 1800.0  # 总时间 s

rho = 820.0  # 密度 kg/m^3
cp = 2600.0  # 比热容 J/(kg·K)
k = 0.36  # 导热系数 W/(m·K)
h = 25.0  # 对流换热系数 W/(m^2·K)


def air_temperature(t):
    """空气温度线性升温近似"""
    return T_air_start + (T_air_end - T_air_start) * (t / t_end)


# ==============================
# 3. 搭建 PINN 网络（简单 MLP）
# ==============================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, r, t):
        # 输入归一化
        r_hat = r / R
        t_hat = t / t_end
        x = torch.cat([r_hat, t_hat], dim=1)

        # 网络输出并映射到真实温度范围
        out = self.net(x)
        T = T0 + (T_air_end - T0) * out
        return T


model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ==============================
# 4. 损失函数定义
# ==============================
def compute_loss(model, n_pde=800, n_ic=150, n_bc=150):
    # ---------- 4.1 PDE 内部点损失 ----------
    r_pde = torch.rand(n_pde, 1, device=device) * R
    r_pde = torch.clamp(r_pde, min=1e-6)  # 防止 1/r 爆炸
    t_pde = torch.rand(n_pde, 1, device=device) * t_end

    r_pde.requires_grad_(True)
    t_pde.requires_grad_(True)
    T_pde = model(r_pde, t_pde)

    dT_dr = torch.autograd.grad(T_pde, r_pde, torch.ones_like(T_pde), create_graph=True)[0]
    dT_dt = torch.autograd.grad(T_pde, t_pde, torch.ones_like(T_pde), create_graph=True)[0]
    d2T_dr2 = torch.autograd.grad(dT_dr, r_pde, torch.ones_like(dT_dr), create_graph=True)[0]

    # 圆柱坐标热传导方程残差
    pde_res = rho * cp * dT_dt - k * (d2T_dr2 + (1.0 / r_pde) * dT_dr)
    pde_scale = rho * cp * (T_air_end - T0) / t_end
    loss_pde = torch.mean((pde_res / pde_scale) ** 2)

    # ---------- 4.2 初始条件损失 ----------
    r_ic = torch.rand(n_ic, 1, device=device) * R
    t_ic = torch.zeros_like(r_ic)
    T_ic = model(r_ic, t_ic)
    loss_ic = torch.mean(((T_ic - T0) / (T_air_end - T0)) ** 2)

    # ---------- 4.3 中心对称边界损失 ----------
    r_center = torch.zeros(n_bc, 1, device=device)
    t_center = torch.rand(n_bc, 1, device=device) * t_end
    r_center.requires_grad_(True)
    T_center = model(r_center, t_center)
    dT_dr_center = torch.autograd.grad(T_center, r_center, torch.ones_like(T_center), create_graph=True)[0]
    center_scale = (T_air_end - T0) / R
    loss_bc_center = torch.mean((dT_dr_center / center_scale) ** 2)

    # ---------- 4.4 表面对流边界损失 ----------
    r_surface = torch.ones(n_bc, 1, device=device) * R
    t_surface = torch.rand(n_bc, 1, device=device) * t_end
    r_surface.requires_grad_(True)
    T_surface = model(r_surface, t_surface)
    dT_dr_surface = torch.autograd.grad(T_surface, r_surface, torch.ones_like(T_surface), create_graph=True)[0]

    T_air_surface = air_temperature(t_surface)
    bc_surface_res = -k * dT_dr_surface - h * (T_surface - T_air_surface)
    surface_scale = h * (T_air_end - T0)
    loss_bc_surface = torch.mean((bc_surface_res / surface_scale) ** 2)

    # ---------- 4.5 总损失 ----------
    loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc_center + 10.0 * loss_bc_surface
    return loss, loss_pde.item(), loss_ic.item(), loss_bc_center.item(), loss_bc_surface.item()


# ==============================
# 5. 开始训练 (已调整为 3000 轮，速度快)
# ==============================
epochs = 3000
start_time = time.time()

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    loss, l_pde, l_ic, l_center, l_surface = compute_loss(model)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | loss = {loss.item():.4f} | pde = {l_pde:.3e} | ic = {l_ic:.3e}")

print(f"训练完成，耗时 {time.time() - start_time:.2f} s")

# ==============================
# 6. 读取传统差分法 result1.xlsx (强制绝对路径，万无一失)
# ==============================
# 获取当前脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "result1.xlsx")

print("正在读取传统差分法结果文件：", filename)
df_temp = pd.read_excel(filename, sheet_name="温度")

# 传统结果的距离点：0, 0.1, ..., 2 cm
r_cm = np.arange(0, 2.001, 0.1)
r_diff = r_cm / 100.0  # 转成 m
T_diff = df_temp.iloc[-1, 1:1 + len(r_cm)].values.astype(float)

# ==============================
# 7. PINN 预测 t = 1800 s 的温度分布
# ==============================
r_test = torch.linspace(0, R, 201, device=device).reshape(-1, 1)
t_test = torch.full_like(r_test, t_end)

with torch.no_grad():
    T_pinn = model(r_test, t_test).cpu().numpy().reshape(-1)

r_test_np = r_test.cpu().numpy().reshape(-1)
T_diff_interp = np.interp(r_test_np, r_diff, T_diff)

# 计算均方误差
mse = np.mean((T_pinn - T_diff_interp) ** 2)
print(f"\n>>> t = 1800 s 时，PINN 与传统差分法温度均方误差 MSE = {mse:.6f} <<<\n")

# ==============================
# 8. 画图对比
# ==============================
plt.figure(figsize=(7, 5))

plt.plot(r_test_np * 100, T_diff_interp, "o-", label="有限差分法 (result1.xlsx)", markersize=3, linewidth=1.5)
plt.plot(r_test_np * 100, T_pinn, "-", label="PINN 预测", linewidth=2)

plt.xlabel("到药材中心的距离 r / cm")
plt.ylabel("温度 T / ℃")
plt.title("t = 1800 s：PINN 与传统差分法温度分布对比")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

save_path = os.path.join(current_dir, "pinn_v1_纯物理.png")
plt.savefig(save_path, dpi=200)
plt.show()

print(f"图像已保存为：{save_path}")