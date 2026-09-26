# check1D.py
# 1D模型检验：网格无关性、时间步长、灵敏度、终点判据对比、消融分析
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from common import tdma

R = 0.02
T0, C0, Cg = 28.0, 2.55, 0.15
hh, hm = 25.0, 8e-7

air = pd.read_excel('附件1.xlsx')
fTa = interp1d(air['时间'], air['温度'], kind='linear', fill_value='extrapolate')
fCa = interp1d(air['时间'], air['水分浓度'], kind='linear', fill_value='extrapolate')
rad = pd.read_excel('附件2.xlsx')
fR = interp1d(rad['时间'], rad['半径'] / 100.0, kind='linear', fill_value='extrapolate')


def get_bc(tNow):
    return (float(fTa(tNow)) if tNow <= 14400 else 50.165,
            float(fCa(tNow)) if tNow <= 14400 else 0.04986)


def run_fixed(Ng=100, dt=60.0, hh=25.0, use_max=True, te=300000):
    dr = R / Ng
    def props(C, T):
        rho = 650 + 128 * C; cp = 1450 + 2736 * C / (C + 1)
        kk = 0.21 + 0.38 * C / (C + 1); TK = T + 273.15
        D = 2.4e-3 * np.exp(-0.45 / np.maximum(C, 1e-6)) * np.exp(-3850 / TK)
        return rho, cp, kk, D
    def updateT(T, C, Ta):
        rho, cp, kk, _ = props(C, T)
        a, b, c, d = np.zeros(Ng+1), np.zeros(Ng+1), np.zeros(Ng+1), np.zeros(Ng+1)
        coef = 4 * kk[0] * dt / (rho[0] * cp[0] * dr * dr)
        b[0], c[0], d[0] = 1 + coef, -coef, T[0]
        for i in range(1, Ng):
            kp = 0.5 * (kk[i] + kk[i+1]); km = 0.5 * (kk[i-1] + kk[i])
            Ap, Am = kp * dt / (rho[i] * cp[i] * dr * dr), km * dt / (rho[i] * cp[i] * dr * dr)
            a[i], c[i] = -Am * (i - 0.5) / i, -Ap * (i + 0.5) / i
            b[i], d[i] = 1 - a[i] - c[i], T[i]
        a[Ng], b[Ng], d[Ng] = -kk[Ng] / dr, kk[Ng] / dr + hh, hh * Ta
        return tdma(a, b, c, d)
    def updateC(C, T, Ca):
        _, _, _, D = props(C, T)
        a, b, c, d = np.zeros(Ng+1), np.zeros(Ng+1), np.zeros(Ng+1), np.zeros(Ng+1)
        coef = 4 * D[0] * dt / (dr * dr)
        b[0], c[0], d[0] = 1 + coef, -coef, C[0]
        for i in range(1, Ng):
            Dp, Dm = 0.5 * (D[i] + D[i+1]), 0.5 * (D[i-1] + D[i])
            Ap, Am = Dp * dt / (dr * dr), Dm * dt / (dr * dr)
            a[i], c[i] = -Am * (i - 0.5) / i, -Ap * (i + 0.5) / i
            b[i], d[i] = 1 - a[i] - c[i], C[i]
        a[Ng], b[Ng], d[Ng] = -D[Ng] / dr, D[Ng] / dr + hm, hm * Ca
        C_new = tdma(a, b, c, d)
        # 【物理限幅】
        C_new[-1] = max(C_new[-1], 0.05)
        return C_new
    T, C = np.full(Ng+1, T0), np.full(Ng+1, C0)
    T1800 = None
    for n in range(1, int(te/dt)+1):
        tNow = n * dt
        Ta, Ca = get_bc(tNow)
        T, C = updateT(T, C, Ta), updateC(C, T, Ca)
        if abs(tNow - 1800) < 1e-6: T1800 = T.copy()
        if use_max:
            done = C.max() < Cg
        else:
            r = np.linspace(0, R, Ng+1); w = 2*r/R**2*dr; w[0]/=2; w[-1]/=2
            done = np.sum(C*w) < Cg
        if done and te > 1800: return tNow/3600, T1800
    return te/3600, T1800


def run_moving(Ng=100, dt=60.0, te=300000):
    dxi = 1.0 / Ng
    def props(C, T):
        rho = 760 + 90*C; cp = 1850 + 2150*C/(C+1)
        kk = 0.12 + 0.20*C/(C+1); TK = T + 273.15
        D = 4.2e-4 * np.exp(-0.30 / np.maximum(C, 1e-6)) * np.exp(-3500/TK)
        return rho, cp, kk, D
    def updateT(T, C, Rt, dR, Ta):
        rho, cp, kk, _ = props(C, T)
        a, b, c, d = np.zeros(Ng+1), np.zeros(Ng+1), np.zeros(Ng+1), np.zeros(Ng+1)
        b[0], c[0], d[0] = 1 + 4*kk[0]*dt/(rho[0]*cp[0]*Rt**2*dxi**2), -4*kk[0]*dt/(rho[0]*cp[0]*Rt**2*dxi**2), T[0]
        for i in range(1, Ng):
            g = (i/Ng)*dR/Rt*(T[i+1]-T[i-1])/(2*dxi)*dt
            kp, km = 0.5*(kk[i]+kk[i+1]), 0.5*(kk[i-1]+kk[i])
            Ap, Am = kp*dt/(rho[i]*cp[i]*Rt**2*dxi**2), km*dt/(rho[i]*cp[i]*Rt**2*dxi**2)
            a[i], c[i] = -Am*(i-0.5)/i, -Ap*(i+0.5)/i
            b[i], d[i] = 1-a[i]-c[i], T[i]+g
        a[Ng], b[Ng], d[Ng] = -kk[Ng]/(Rt*dxi), kk[Ng]/(Rt*dxi)+hh, hh*Ta
        return tdma(a, b, c, d)
    def updateC(C, T, Rt, dR, Ca):
        _, _, _, D = props(C, T)
        a, b, c, d = np.zeros(Ng+1), np.zeros(Ng+1), np.zeros(Ng+1), np.zeros(Ng+1)
        b[0], c[0], d[0] = 1 + 4*D[0]*dt/(Rt**2*dxi**2), -4*D[0]*dt/(Rt**2*dxi**2), C[0]
        for i in range(1, Ng):
            g = (i/Ng)*dR/Rt*(C[i+1]-C[i-1])/(2*dxi)*dt
            Dp, Dm = 0.5*(D[i]+D[i+1]), 0.5*(D[i-1]+D[i])
            Ap, Am = Dp*dt/(Rt**2*dxi**2), Dm*dt/(Rt**2*dxi**2)
            a[i], c[i] = -Am*(i-0.5)/i, -Ap*(i+0.5)/i
            b[i], d[i] = 1-a[i]-c[i], C[i]+g
        a[Ng], b[Ng], d[Ng] = -D[Ng]/(Rt*dxi), D[Ng]/(Rt*dxi)+hm, hm*Ca
        C_new = tdma(a, b, c, d)
        # 【物理限幅】
        C_new[-1] = max(C_new[-1], 0.05)
        return C_new
    T, C = np.full(Ng+1, T0), np.full(Ng+1, C0)
    for n in range(1, int(te/dt)+1):
        tNow = n*dt; Rt, Rnext = float(fR(tNow)), float(fR(tNow+dt))
        dR = (Rnext-Rt)/dt; Ta, Ca = get_bc(tNow)
        T, C = updateT(T, C, Rt, dR, Ta), updateC(C, T, Rt, dR, Ca)
        if C.max() < Cg: return tNow/3600
    return te/3600


if __name__ == "__main__":
    print("="*50); print("【1】网格无关性"); print("="*50)
    for N in [50, 100, 200]:
        t, _ = run_fixed(Ng=N, dt=60.0)
        print(f"N={N:3d}: 烘干时间 {t:.2f} h")
    print("="*50); print("【2】时间步长收敛性"); print("="*50)
    for dt in [120, 60, 30]:
        t, _ = run_fixed(Ng=100, dt=dt)
        print(f"dt={dt:3d}s: 烘干时间 {t:.2f} h")
    print("="*50); print("【3】灵敏度分析 (1800s表面温度)"); print("="*50)
    for h_val in [22.5, 25.0, 27.5]:
        _, T_1800 = run_fixed(Ng=100, dt=60.0, hh=h_val, te=1800)
        print(f"h={h_val:4.1f}: {T_1800[-1]:.4f} ℃")
    print("="*50); print("【4】终点判据对比"); print("="*50)
    t_max, _ = run_fixed(Ng=100, dt=60.0, use_max=True)
    t_avg, _ = run_fixed(Ng=100, dt=60.0, use_max=False)
    print(f"全域最大判据: {t_max:.2f} h")
    print(f"平均含水率判据: {t_avg:.2f} h")
    print("="*50); print("【5】尺寸收缩消融分析"); print("="*50)
    t_fixed, _ = run_fixed(Ng=100, dt=60.0)
    t_move = run_moving(Ng=100, dt=60.0)
    print(f"对照组(固定半径, 附录3): {t_fixed:.2f} h")
    print(f"完整组(动态收缩, 附录4): {t_move:.2f} h")
    print(f"缩短: {abs(t_fixed-t_move):.2f} h ({(abs(t_fixed-t_move)/t_fixed*100):.2f}%)")