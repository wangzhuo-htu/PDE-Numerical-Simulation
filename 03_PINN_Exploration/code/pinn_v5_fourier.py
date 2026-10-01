# -*- coding: utf-8 -*-
"""
基于 PINN 的圆柱形药材预热阶段一维径向热传导
【第五版：傅里叶特征 + 宽网络 + 学习率衰减】
核心改进：解决神经网络“频谱偏置”问题，强行学会空间梯度！
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
R = 0.02
T0 = 28.0
T_air_start = 28.0
T_air_end = 41.513
t_end = 1800.0
rho = 820.0
cp = 2600.0
k = 0.36
h = 25.0


def air_temperature(t):
    return T_air_start + (T_air_end - T_air_start) * (t / t_end)


# ==============================
# 3. 读取完整时空数据
# ==============================
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "result1.xlsx")
print("正在读取传统差分法结果文件：", filename)

df_temp = pd.read_excel(filename, sheet_name="温度")
t_data = df_temp.iloc[:, 0].values.astype(np.float32)
r_data_cm = df_temp.columns[1:].values.astype(np.float32)
r_data = r_data_cm / 100.0
T_data_mat = df_temp.iloc[:, 1:].values.astype(np.float32)

r_flat = np.tile(r_data, len(t_data))
t_flat = np.repeat(t_data, len(r_data))
T_flat = T_data_mat.flatten()

r_gt_tensor = torch.tensor(r_flat, dtype=torch.float32, device=device).reshape(-1, 1)
t_gt_tensor = torch.tensor(t_flat, dtype=torch.float32, device=device).reshape(-1, 1)
T_gt_tensor = torch.tensor(T_flat, dtype=torch.float32, device=device).reshape(-1, 1)

print(f"成功读取数据：共 {len(r_flat)} 个时空点")


# ==============================
# 4. 搭建 PINN 网络（引入傅里叶特征）
# ==============================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # 输入层由原来的 2 维变成了 6 维！
        # 因为我们对 r 进行了傅里叶特征映射
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(),  # 加宽到 64
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, r, t):
        r_hat = r / R
        t_hat = t / t_end

        # 傅里叶特征映射：给网络戴上“高频率眼镜”
        # 把 r 映射成 sin(pi*r), cos(pi*r), sin(2*pi*r), cos(2*pi*r)
        r_feat = torch.cat([
            r_hat,
            torch.sin(np.pi * r_hat),
            torch.cos(np.pi * r_hat),
            torch.sin(2 * np.pi * r_hat),
            torch.cos(2 * np.pi * r_hat)
        ], dim=1)

        # 最终输入是 [r_feat, t_hat]，共 6 维
        x = torch.cat([r_feat, t_hat], dim=1)

        out = self.net(x)
        return T0 + (T_air_end - T0) * out


model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 学习率衰减：每 3000 轮学习率减半，让最后阶段精细收敛
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)


# ==============================
# 5. 损失函数
# ==============================
def compute_loss(model, n_pde=2000, n_ic=200, n_bc=200, n_data=8000):
    # 5.1 PDE 物理损失
    r_pde = torch.rand(n_pde, 1, device=device) * R
    r_pde = torch.clamp(r_pde, min=1e-6)
    t_pde = torch.rand(n_pde, 1, device=device) * t_end
    r_pde.requires_grad_(True)
    t_pde.requires_grad_(True)
    T_pde = model(r_pde, t_pde)

    dT_dr = torch.autograd.grad(T_pde, r_pde, torch.ones_like(T_pde), create_graph=True)[0]
    dT_dt = torch.autograd.grad(T_pde, t_pde, torch.ones_like(T_pde), create_graph=True)[0]
    d2T_dr2 = torch.autograd.grad(dT_dr, r_pde, torch.ones_like(dT_dr), create_graph=True)[0]

    pde_res = rho * cp * dT_dt - k * (d2T_dr2 + (1.0 / r_pde) * dT_dr)
    loss_pde = torch.mean((pde_res / (rho * cp * (T_air_end - T0) / t_end)) ** 2)

    # 5.2 初始条件损失
    r_ic = torch.rand(n_ic, 1, device=device) * R
    t_ic = torch.zeros_like(r_ic)
    loss_ic = torch.mean(((model(r_ic, t_ic) - T0) / (T_air_end - T0)) ** 2)

    # 5.3 中心对称边界损失
    r_center = torch.zeros(n_bc, 1, device=device)
    t_center = torch.rand(n_bc, 1, device=device) * t_end
    r_center.requires_grad_(True)
    T_center = model(r_center, t_center)
    dT_dr_center = torch.autograd.grad(T_center, r_center, torch.ones_like(T_center), create_graph=True)[0]
    loss_bc_center = torch.mean((dT_dr_center / ((T_air_end - T0) / R)) ** 2)

    # 5.4 表面对流边界损失
    r_surface = torch.ones(n_bc, 1, device=device) * R
    t_surface = torch.rand(n_bc, 1, device=device) * t_end
    r_surface.requires_grad_(True)
    T_surface = model(r_surface, t_surface)
    dT_dr_surface = torch.autograd.grad(T_surface, r_surface, torch.ones_like(T_surface), create_graph=True)[0]
    bc_surface_res = -k * dT_dr_surface - h * (T_surface - air_temperature(t_surface))
    loss_bc_surface = torch.mean((bc_surface_res / (h * (T_air_end - T0))) ** 2)

    # 5.5 全时空数据监督损失（增加采样点）
    idx = torch.randint(0, len(r_gt_tensor), (n_data,))
    T_pred = model(r_gt_tensor[idx], t_gt_tensor[idx])
    loss_data = torch.mean(((T_pred - T_gt_tensor[idx]) / (T_air_end - T0)) ** 2)

    # 总损失：数据权重依然给 50，但因为有傅里叶特征，它现在真的能学弯了！
    loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc_center + 10.0 * loss_bc_surface + 50.0 * loss_data

    return loss, loss_pde.item(), loss_ic.item(), loss_data.item()


# ==============================
# 6. 训练 (15000轮，带有学习率衰减)
# ==============================
epochs = 15000
start_time = time.time()

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    loss, l_pde, l_ic, l_data = compute_loss(model)
    loss.backward()
    optimizer.step()

    scheduler.step()  # 更新学习率

    if epoch % 1500 == 0:
        print(
            f"Epoch {epoch:5d} | lr = {scheduler.get_last_lr()[0]:.1e} | loss = {loss.item():.4f} | pde = {l_pde:.3e} | data = {l_data:.3e}")

print(f"训练完成，耗时 {time.time() - start_time:.2f} s")

# ==============================
# 7. 预测并计算误差
# ==============================
r_test = torch.linspace(0, R, 201, device=device).reshape(-1, 1)
t_test = torch.full_like(r_test, t_end)

with torch.no_grad():
    T_pinn = model(r_test, t_test).cpu().numpy().reshape(-1)

r_test_np = r_test.cpu().numpy().reshape(-1)
T_diff_final = T_data_mat[-1, :]
T_diff_interp = np.interp(r_test_np, r_data, T_diff_final)

mse = np.mean((T_pinn - T_diff_interp) ** 2)
print(f"\n>>> t = 1800 s 时，PINN 与传统差分法温度均方误差 MSE = {mse:.6f} <<<\n")

# ==============================
# 8. 画图对比
# ==============================
plt.figure(figsize=(7, 5))
plt.plot(r_test_np * 100, T_diff_interp, "o-", label="有限差分法 (Ground Truth)", markersize=4, linewidth=1.5)
plt.plot(r_test_np * 100, T_pinn, "-", label="PINN 预测 (傅里叶特征版)", linewidth=2.5)

plt.xlabel("到药材中心的距离 r / cm")
plt.ylabel("温度 T / ℃")
plt.title("t = 1800 s：傅里叶特征 PINN 对比")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

save_path = os.path.join(current_dir, "pinn_vs_diff_t1800_v5.png")
plt.savefig(save_path, dpi=200)
plt.show()

print(f"图像已保存为：{save_path}")