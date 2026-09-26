# 问题三：全域终点判据，确定烘干时间
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from common import tdma

R = 0.02
Ng = 100
dr = R / Ng
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

nStep = int(tmax / dt)


def props(C, T):
    rho = 650 + 128 * C
    cp = 1450 + 2736 * C / (C + 1)
    kk = 0.21 + 0.38 * C / (C + 1)
    TK = T + 273.15
    D = 2.4e-3 * np.exp(-0.45 / np.maximum(C, 1e-6)) * np.exp(-3850 / TK)
    return rho, cp, kk, D


def updateT(T, C, Ta):
    rho, cp, kk, _ = props(C, T)
    a = np.zeros(Ng + 1); b = np.zeros(Ng + 1)
    c = np.zeros(Ng + 1); d = np.zeros(Ng + 1)
    coef = 4 * kk[0] * dt / (rho[0] * cp[0] * dr * dr)
    b[0] = 1 + coef; c[0] = -coef; d[0] = T[0]
    for i in range(1, Ng):
        kp = 0.5 * (kk[i] + kk[i + 1]); km = 0.5 * (kk[i - 1] + kk[i])
        Ap = kp * dt / (rho[i] * cp[i] * dr * dr)
        Am = km * dt / (rho[i] * cp[i] * dr * dr)
        a[i] = -Am * (i - 0.5) / i; c[i] = -Ap * (i + 0.5) / i
        b[i] = 1 - a[i] - c[i]; d[i] = T[i]
    a[Ng] = -kk[Ng] / dr; b[Ng] = kk[Ng] / dr + hh; d[Ng] = hh * Ta
    return tdma(a, b, c, d)


def updateC(C, T, Ca):
    _, _, _, D = props(C, T)
    a = np.zeros(Ng + 1); b = np.zeros(Ng + 1)
    c = np.zeros(Ng + 1); d = np.zeros(Ng + 1)
    coef = 4 * D[0] * dt / (dr * dr)
    b[0] = 1 + coef; c[0] = -coef; d[0] = C[0]
    for i in range(1, Ng):
        Dp = 0.5 * (D[i] + D[i + 1]); Dm = 0.5 * (D[i - 1] + D[i])
        Ap = Dp * dt / (dr * dr); Am = Dm * dt / (dr * dr)
        a[i] = -Am * (i - 0.5) / i; c[i] = -Ap * (i + 0.5) / i
        b[i] = 1 - a[i] - c[i]; d[i] = C[i]
    a[Ng] = -D[Ng] / dr; b[Ng] = D[Ng] / dr + hm; d[Ng] = hm * Ca
    C_new = tdma(a, b, c, d)
    # 【物理限幅】表面水分不低于空气平衡浓度，避免数值下冲
    C_new[-1] = max(C_new[-1], 0.05)
    return C_new


T = np.full(Ng + 1, T0); C = np.full(Ng + 1, C0)
Ts = [T.copy()]; Cs = [C.copy()]; tt = [0.0]
tEnd = None

for n in range(1, nStep + 1):
    tNow = n * dt
    if tNow <= 14400:
        Ta = float(fTa(tNow))
        Ca = float(fCa(tNow))
    else:
        Ta = 50.165
        Ca = 0.04986

    T = updateT(T, C, Ta)
    C = updateC(C, T, Ca)
    tt.append(tNow); Ts.append(T.copy()); Cs.append(C.copy())
    if n % 60 == 0:
        print(f'  t = {tNow/3600:.2f} h, maxC = {C.max():.4f}')
    if C.max() < Cg:
        tEnd = tNow
        print(f'  达标时间 t = {tNow} s = {tNow / 3600:.2f} h')
        break

Cs = np.array(Cs); tt = np.array(tt)
rIn = np.linspace(0, R, Ng + 1); rOut = np.arange(0, 2.001, 0.1); rOutM = rOut / 100.0
Co = np.zeros((len(tt), len(rOut)))
for n in range(len(tt)):
    Co[n, :] = np.interp(rOutM, rIn, Cs[n, :])

colNames = [f'{r:.1f}' for r in rOut]
dfC = pd.DataFrame(Co, columns=colNames); dfC.insert(0, '时间', tt)
dfC.to_excel('result3.xlsx', sheet_name='水分浓度', index=False, float_format='%.4f')

if tEnd is not None:
    tShow = list(np.arange(6, tEnd / 3600, 6)) + [tEnd / 3600]
    rIdx = [0, 5, 10, 15, 20]; rCm = [0, 0.5, 1, 1.5, 2]
    rows = []
    for h in tShow:
        idx = int(h * 3600 / dt); idx = min(idx, len(tt) - 1)
        rows.append([Co[idx, i] for i in rIdx])
    dfC5 = pd.DataFrame(np.array(rows).round(4), index=[f'{h:.2f}' for h in tShow], columns=rCm)
    dfC5.index.name = '时间/h'
    dfC5.to_excel('表5_水分浓度_烘干过程.xlsx')

print('solve3 完成')