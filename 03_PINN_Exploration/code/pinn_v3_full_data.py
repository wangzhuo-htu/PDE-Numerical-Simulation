# -*- coding: utf-8 -*-
"""
基于 PINN 的圆柱形药材预热阶段一维径向热传导
【第三版：全时空数据 + 物理联合驱动】
核心改进：读取 result1.xlsx 的全部时空数据，作为数据监督！
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

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================
# 2. 物理参数
# ==============================
R = 0.02                 # 半径 m
T0 = 28.0                # 初始温度 ℃
T_air_start = 28.0       # 初始空气温度 ℃
T_air_end = 41.513       # 1800s空气温度 ℃
t_end = 1800.0           # 总时间 s
rho = 820.0              # 密度
cp = 2600.0              # 比热容
k = 0.36                 # 导热系数
h = 25.0                 # 对流换热系数

def air_temperature(t):
    """空气温度线性升温近似"""
    return T_air_start + (T_air_end - T_air_start) * (t / t_end)

# ==============================
# 3. 读取完整时空数据 (核心修改！)
# ==============================
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "result1.xlsx")
print("正在读取传统差分法结果文件：", filename)

df_temp = pd.read_excel(filename, sheet_name="温度")

# 提取时间列 (第一列) 和 空间坐标 (第一行，去掉"时间"两个字)
t_data = df_temp.iloc[:, 0].values.astype(np.float32)  # 0, 1, 2, ..., 1800
r_data_cm = df_temp.columns[1:].values.astype(np.float32)  # 0, 0.1, ..., 2.0
r_data = r_data_cm / 100.0  # 转成米

# 提取温度矩阵 (去掉第一列时间)
T_data_mat = df_temp.iloc[:, 1:].values.astype(np.float32)  # 形状 (1801, 21)

# 把二维矩阵拉平成 1D 数组，方便随机采样
r_flat = np.tile(r_data, len(t_data))       # 重复 21 次
t_flat = np.repeat(t_data, len(r_data))     # 每个时间点重复 21 次
T_flat = T_data_mat.flatten()               # 展平所有温度数据

# 转成 PyTorch 张量
r_gt_tensor = torch.tensor(r_flat, dtype=torch.float32, device=device).reshape(-1, 1)
t_gt_tensor = torch.tensor(t_flat, dtype=torch.float32, device=device).reshape(-1, 1)
T_gt_tensor = torch.tensor(T_flat, dtype=torch.float32, device=device).reshape(-1, 1)

print(f"成功读取数据：共 {len(r_flat)} 个时空点")

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
# 5. 损失函数 (加入全时空数据损失)
# ==============================
def compute_loss(model, n_pde=1000, n_ic=200, n_bc=200, n_data=1000):
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

    # ---------- 5.5 全时空数据监督损失 (核心改进！) ----------
    # 随机从整个数据集中抽取 n_data 个时空点来“补课”
    idx = torch.randint(0, len(r_gt_tensor), (n_data,))
    r_batch = r_gt_tensor[idx]
    t_batch = t_gt_tensor[idx]
    T_batch = T_gt_tensor[idx]

    T_pred = model(r_batch, t_batch)
    loss_data = torch.mean(((T_pred - T_batch) / (T_air_end - T0)) ** 2)

    # 总损失：给数据损失非常高的权重 (100.0)，让它必须听话！
    loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc_center + 10.0 * loss_bc_surface + 100.0 * loss_data

    return loss, loss_pde.item(), loss_ic.item(), loss_data.item()

# ==============================
# 6. 开始训练 (5000轮，速度很快)
# ==============================
epochs = 5000
start_time = time.time()

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    loss, l_pde, l_ic, l_data = compute_loss(model)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | loss = {loss.item():.4f} | pde = {l_pde:.3e} | ic = {l_ic:.3e} | data = {l_data:.3e}")

print(f"训练完成，耗时 {time.time() - start_time:.2f} s")

# ==============================
# 7. 预测并计算误差 (只看 t=1800s)
# ==============================
r_test = torch.linspace(0, R, 201, device=device).reshape(-1, 1)
t_test = torch.full_like(r_test, t_end)  # 预测 t=1800s 的结果

with torch.no_grad():
    T_pinn = model(r_test, t_test).cpu().numpy().reshape(-1)

r_test_np = r_test.cpu().numpy().reshape(-1)

# 取出 t=1800s 的真实数据用于对比
T_diff_final = T_data_mat[-1, :]  # 最后一行
T_diff_interp = np.interp(r_test_np, r_data, T_diff_final)

mse = np.mean((T_pinn - T_diff_interp) ** 2)
print(f"\n>>> t = 1800 s 时，PINN 与传统差分法温度均方误差 MSE = {mse:.6f} <<<\n")

# ==============================
# 8. 画图对比
# ==============================
plt.figure(figsize=(7, 5))
plt.plot(r_test_np * 100, T_diff_interp, "o-", label="有限差分法 (Ground Truth)", markersize=4, linewidth=1.5)
plt.plot(r_test_np * 100, T_pinn, "-", label="PINN 预测 (全时空数据驱动)", linewidth=2.5)

plt.xlabel("到药材中心的距离 r / cm")
plt.ylabel("温度 T / ℃")
plt.title("t = 1800 s：全时空数据监督后的 PINN 对比")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

save_path = os.path.join(current_dir, "pinn_vs_diff_t1800_v3.png")
plt.savefig(save_path, dpi=200)
plt.show()

print(f"图像已保存为：{save_path}")