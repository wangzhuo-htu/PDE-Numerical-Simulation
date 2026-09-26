# 问题二：变物性传热传质模型
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from common import tdma

R = 0.02
Ng = 100
dr = R / Ng
dt = 1.0
te = 10800

T0 = 28.0
C0 = 2.55
hh = 25.0
hm = 8e-7

air = pd.read_excel('附件1.xlsx')
fTa = interp1d(air['时间'], air['温度'], kind='linear', fill_value='extrapolate')
fCa = interp1d(air['时间'], air['水分浓度'], kind='linear', fill_value='extrapolate')

nStep = int(te / dt)
tt = np.arange(0, nStep + 1) * dt

def props(C, T):
    rho = 650 + 128 * C
    cp = 1450 + 2736 * C / (C + 1)
    kk = 0.21 + 0.38 * C / (C + 1)
    TK = T + 273.15
    # 修正：根据附件3经验公式，系数是 0.45
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
    return tdma(a, b, c, d)

T = np.full(Ng + 1, T0); C = np.full(Ng + 1, C0)
Ts = np.zeros((nStep + 1, Ng + 1)); Cs = np.zeros((nStep + 1, Ng + 1))
Ts[0] = T; Cs[0] = C

for n in range(nStep):
    tNow = n * dt
    # 14400s 之后使用恒温外推
    if tNow <= 14400:
        Ta = float(fTa(tNow))
        Ca = float(fCa(tNow))
    else:
        Ta = 50.165
        Ca = 0.04986

    T = updateT(T, C, Ta)
    C = updateC(C, T, Ca)
    Ts[n + 1] = T; Cs[n + 1] = C
    if n % 1800 == 0: print('  n =', n)

rIn = np.linspace(0, R, Ng + 1); rOut = np.arange(0, 2.001, 0.1); rOutM = rOut / 100.0
To = np.zeros((nStep + 1, len(rOut))); Co = np.zeros((nStep + 1, len(rOut)))
for n in range(nStep + 1):
    To[n, :] = np.interp(rOutM, rIn, Ts[n, :]); Co[n, :] = np.interp(rOutM, rIn, Cs[n, :])

colNames = [f'{r:.1f}' for r in rOut]
dfT = pd.DataFrame(To, columns=colNames); dfT.insert(0, '时间', tt)
dfC = pd.DataFrame(Co, columns=colNames); dfC.insert(0, '时间', tt)
with pd.ExcelWriter('result2.xlsx') as writer:
    dfT.to_excel(writer, sheet_name='温度', index=False, float_format='%.4f')
    dfC.to_excel(writer, sheet_name='水分浓度', index=False, float_format='%.4f')

tShow = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]; rIdx = [0, 5, 10, 15, 20]; rCm = [0, 0.5, 1, 1.5, 2]
rowsT = []; rowsC = []
for h in tShow:
    idx = int(h * 3600)
    rowsT.append([To[idx, i] for i in rIdx]); rowsC.append([Co[idx, i] for i in rIdx])
dfT3 = pd.DataFrame(np.array(rowsT).round(4), index=tShow, columns=rCm); dfT3.index.name = '时间/h'
dfC3 = pd.DataFrame(np.array(rowsC).round(4), index=tShow, columns=rCm); dfC3.index.name = '时间/h'
dfT3.to_excel('表3_温度_3h.xlsx'); dfC3.to_excel('表4_水分浓度_3h.xlsx')

print('3h 表面温度/水分:', To[-1, -1], Co[-1, -1])
print('3h 中心温度/水分:', To[-1, 0], Co[-1, 0])
print('solve2 完成')