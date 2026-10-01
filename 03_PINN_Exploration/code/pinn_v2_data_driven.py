# -*- coding: utf-8 -*-
"""
基于 PINN 的圆柱形药材预热阶段一维径向热传导
【第二版：数据 + 物理联合驱动】
核心改进：除了物理方程，还加入了传统差分法结果的“数据监督”
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
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 画图字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================
# 2. 物理参数
# ==============================
R = 0.02  # 半径 m
T0 = 28.0  # 初始温度 ℃
T_air_start = 28.0  # 初始空气温度 ℃
T_air_end = 41.513  # 1800s空气温度 ℃
t_end = 1800.0  # 总时间 s
rho = 820.0  # 密度
cp = 2600.0  # 比热容
k = 0.36  # 导热系数
h = 25.0  # 对流换热系数


def air_temperature(t):
    """空气温度线性升温近似"""
    return T_air_start + (T_air_end - T_air_start) * (t / t_end)


# ==============================
# 3. 读取传统差分法数据，提取“数据监督”项
# ==============================
# 获取当前脚本所在的绝对路径，保证能找到文件
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "result1.xlsx")
print("正在读取传统差分法结果文件：", filename)

df_temp = pd.read_excel(filename, sheet_name="温度")
r_cm = np.arange(0, 2.001, 0.1)
r_diff = r_cm / 100.0
T_diff = df_temp.iloc[-1, 1:1 + len(r_cm)].values.astype(float)

# 转成 PyTorch 张量，用于训练时给网络“补课”
# 注意：这里把数据放在 device 上，方便直接计算损失
r_gt_tensor = torch.tensor(r_diff, dtype=torch.float32, device=device).reshape(-1, 1)
T_gt_tensor = torch.tensor(T_diff, dtype=torch.float32, device=device).reshape(-1, 1)


# ==============================
# 4. 搭建 PINN 网络
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
        r_hat = r / R
        t_hat = t / t_end
        x = torch.cat([r_hat, t_hat], dim=1)

        out = self.net(x)
        # 映射到真实温度范围 [T0, T_air_end]
        T = T0 + (T_air_end - T0) * out
        return T


model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ==============================
# 5. 损失函数（核心：加入了数据监督 loss_data）
# ==============================
def compute_loss(model, n_pde=1000, n_ic=200, n_bc=200):
    # ---------- 5.1 PDE 物理损失 ----------
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

    # ---------- 5.2 初始条件损失 ----------
    r_ic = torch.rand(n_ic, 1, device=device) * R
    t_ic = torch.zeros_like(r_ic)
    T_ic = model(r_ic, t_ic)
    loss_ic = torch.mean(((T_ic - T0) / (T_air_end - T0)) ** 2)

    # ---------- 5.3 中心对称边界损失 ----------
    r_center = torch.zeros(n_bc, 1, device=device)
    t_center = torch.rand(n_bc, 1, device=device) * t_end
    r_center.requires_grad_(True)
    T_center = model(r_center, t_center)
    dT_dr_center = torch.autograd.grad(T_center, r_center, torch.ones_like(T_center), create_graph=True)[0]
    center_scale = (T_air_end - T0) / R
    loss_bc_center = torch.mean((dT_dr_center / center_scale) ** 2)

    # ---------- 5.4 表面对流边界损失 ----------
    r_surface = torch.ones(n_bc, 1, device=device) * R
    t_surface = torch.rand(n_bc, 1, device=device) * t_end
    r_surface.requires_grad_(True)
    T_surface = model(r_surface, t_surface)
    dT_dr_surface = torch.autograd.grad(T_surface, r_surface, torch.ones_like(T_surface), create_graph=True)[0]

    T_air_surface = air_temperature(t_surface)
    bc_surface_res = -k * dT_dr_surface - h * (T_surface - T_air_surface)
    surface_scale = h * (T_air_end - T0)
    loss_bc_surface = torch.mean((bc_surface_res / surface_scale) ** 2)

    # ---------- 5.5 数据监督损失 (第二版新增！) ----------
    # 强行让网络在 t=1800s 时，去拟合传统差分法给出的标准答案
    t_gt = torch.full_like(r_gt_tensor, t_end)
    T_gt_pred = model(r_gt_tensor, t_gt)
    loss_data = torch.mean(((T_gt_pred - T_gt_tensor) / (T_air_end - T0)) ** 2)

    # 总损失：给数据损失非常高的权重 (100.0)，让网络必须向标准答案学习
    loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc_center + 10.0 * loss_bc_surface + 100.0 * loss_data

    return loss, loss_pde.item(), loss_ic.item(), loss_data.item()


# ==============================
# 6. 开始训练
# ==============================
# 轮数加到10000，让网络有足够时间收敛
epochs = 10000
start_time = time.time()

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    loss, l_pde, l_ic, l_data = compute_loss(model)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch:5d} | loss = {loss.item():.4f} | pde = {l_pde:.3e} | ic = {l_ic:.3e} | data = {l_data:.3e}")

print(f"训练完成，耗时 {time.time() - start_time:.2f} s")

# ==============================
# 7. 预测并计算误差
# ==============================
r_test = torch.linspace(0, R, 201, device=device).reshape(-1, 1)
t_test = torch.full_like(r_test, t_end)

with torch.no_grad():
    T_pinn = model(r_test, t_test).cpu().numpy().reshape(-1)

r_test_np = r_test.cpu().numpy().reshape(-1)
T_diff_interp = np.interp(r_test_np, r_diff, T_diff)

mse = np.mean((T_pinn - T_diff_interp) ** 2)
print(f"\n>>> t = 1800 s 时，PINN 与传统差分法温度均方误差 MSE = {mse:.6f} <<<\n")

# ==============================
# 8. 画图对比
# ==============================
plt.figure(figsize=(7, 5))
plt.plot(r_test_np * 100, T_diff_interp, "o-", label="有限差分法 (Ground Truth)", markersize=4, linewidth=1.5)
plt.plot(r_test_np * 100, T_pinn, "-", label="PINN 预测 (数据+物理驱动)", linewidth=2.5)

plt.xlabel("到药材中心的距离 r / cm")
plt.ylabel("温度 T / ℃")
plt.title("t = 1800 s：加入数据监督后的 PINN 对比")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

save_path = os.path.join(current_dir, "pinn_vs_diff_t1800_v2.png")
plt.savefig(save_path, dpi=200)
plt.show()

print(f"图像已保存为：{save_path}")