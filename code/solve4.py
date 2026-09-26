# 问题四：移动边界，半径随脱水收缩
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from common import tdma

R0 = 0.02
Ng = 100
dxi = 1.0 / Ng
dt = 60.0
tmax = 300000

T0 = 28.0
C0 = 2.55
Cg = 0.15
hh = 25.0
hm = 8e-7

air = pd.read_excel('附件1.xlsx')
fTa = interp1d(air['时间'], air['温度'], kind='linear', fill_value='extrapolate')
fCa = interp1d(air['时间'], air['水分浓度'], kind='linear', fill_value='extrapolate')

rad = pd.read_excel('附件2.xlsx')
fR = interp1d(rad['时间'], rad['半径'] / 100.0, kind='linear', fill_value='extrapolate')

nStep = int(tmax / dt)


def props(C, T):
    rho = 760 + 90 * C
    cp = 1850 + 2150 * C / (C + 1)
    kk = 0.12 + 0.20 * C / (C + 1)
    TK = T + 273.15
    D = 4.2e-4 * np.exp(-0.30 / np.maximum(C, 1e-6)) * np.exp(-3500 / TK)
    return rho, cp, kk, D


def updateT(T, C, R, dR, Ta):
    rho, cp, kk, _ = props(C, T)
    a = np.zeros(Ng + 1); b = np.zeros(Ng + 1)
    c = np.zeros(Ng + 1); d = np.zeros(Ng + 1)
    coef = 4 * kk[0] * dt / (rho[0] * cp[0] * R * R * dxi * dxi)
    b[0] = 1 + coef; c[0] = -coef; d[0] = T[0]
    for i in range(1, Ng):
        g = (i / Ng) * dR / R * (T[i + 1] - T[i - 1]) / (2 * dxi) * dt
        kp = 0.5 * (kk[i] + kk[i + 1]); km = 0.5 * (kk[i - 1] + kk[i])
        Ap = kp * dt / (rho[i] * cp[i] * R * R * dxi * dxi)
        Am = km * dt / (rho[i] * cp[i] * R * R * dxi * dxi)
        a[i] = -Am * (i - 0.5) / i; c[i] = -Ap * (i + 0.5) / i
        b[i] = 1 - a[i] - c[i]; d[i] = T[i] + g
    a[Ng] = -kk[Ng] / (R * dxi); b[Ng] = kk[Ng] / (R * dxi) + hh
    d[Ng] = hh * Ta
    return tdma(a, b, c, d)


def updateC(C, T, R, dR, Ca):
    _, _, _, D = props(C, T)
    a = np.zeros(Ng + 1); b = np.zeros(Ng + 1)
    c = np.zeros(Ng + 1); d = np.zeros(Ng + 1)
    coef = 4 * D[0] * dt / (R * R * dxi * dxi)
    b[0] = 1 + coef; c[0] = -coef; d[0] = C[0]
    for i in range(1, Ng):
        g = (i / Ng) * dR / R * (C[i + 1] - C[i - 1]) / (2 * dxi) * dt
        Dp = 0.5 * (D[i] + D[i + 1]); Dm = 0.5 * (D[i - 1] + D[i])
        Ap = Dp * dt / (R * R * dxi * dxi); Am = Dm * dt / (R * R * dxi * dxi)
        a[i] = -Am * (i - 0.5) / i; c[i] = -Ap * (i + 0.5) / i
        b[i] = 1 - a[i] - c[i]; d[i] = C[i] + g
    a[Ng] = -D[Ng] / (R * dxi); b[Ng] = D[Ng] / (R * dxi) + hm
    d[Ng] = hm * Ca
    C_new = tdma(a, b, c, d)
    # 【物理限幅】表面水分不低于空气平衡浓度
    C_new[-1] = max(C_new[-1], 0.05)
    return C_new


T = np.full(Ng + 1, T0); C = np.full(Ng + 1, C0)
Ts = [T.copy()]; Cs = [C.copy()]; Rs = [R0]; tt = [0.0]
tEnd = None

for n in range(1, nStep + 1):
    tNow = n * dt
    Rnow = float(fR(tNow)); Rnext = float(fR(tNow + dt))
    dR = (Rnext - Rnow) / dt

    if tNow <= 14400:
        Ta = float(fTa(tNow))
        Ca = float(fCa(tNow))
    else:
        Ta = 50.165
        Ca = 0.04986

    T = updateT(T, C, Rnow, dR, Ta)
    C = updateC(C, T, Rnow, dR, Ca)
    tt.append(tNow); Ts.append(T.copy()); Cs.append(C.copy()); Rs.append(Rnow)
    if n % 60 == 0:
        print(f'  t = {tNow/3600:.2f} h, R = {Rnow*100:.3f} cm, maxC = {C.max():.4f}')
    if C.max() < Cg:
        tEnd = tNow
        print(f'  达标时间 t = {tNow} s = {tNow / 3600:.2f} h')
        break

Cs = np.array(Cs); Rs = np.array(Rs); tt = np.array(tt)
rOut = np.arange(0, 2.001, 0.1)
Co = np.zeros((len(tt), len(rOut) + 1))
for n in range(len(tt)):
    R = Rs[n]
    for j, r in enumerate(rOut):
        rM = r / 100.0
        if rM <= R:
            x = rM / R
            i0 = int(np.floor(x * Ng)); i1 = min(i0 + 1, Ng)
            w = x * Ng - i0
            Co[n, j] = (1 - w) * Cs[n, i0] + w * Cs[n, i1]
        else:
            Co[n, j] = np.nan
    Co[n, -1] = Cs[n, -1]

colNames = [f'{r:.1f}' for r in rOut] + ['表面']
dfC = pd.DataFrame(Co, columns=colNames); dfC.insert(0, '时间', tt)
dfC.to_excel('result4.xlsx', sheet_name='水分浓度', index=False, float_format='%.4f')

if tEnd is not None:
    tShow = list(np.arange(6, tEnd / 3600, 6)) + [tEnd / 3600]
    rows = []
    for h in tShow:
        idx = int(h * 3600 / dt); idx = min(idx, len(tt) - 1)
        rows.append(Co[idx])
    dfC6 = pd.DataFrame(np.array(rows).round(4), index=[f'{h:.2f}' for h in tShow], columns=colNames)
    dfC6.index.name = '时间/h'
    dfC6.to_excel('表6_水分浓度_收缩模型.xlsx')

print('solve4 完成')
