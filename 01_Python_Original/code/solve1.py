# 问题一：预热平衡阶段温度与水分浓度变化模型
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from common import tdma

R = 0.02
Ng = 100
dr = R / Ng
dt = 1.0
te = 1800

T0 = 28.0
C0 = 2.55

rho = 820.0
cp = 2600.0
kk = 0.36
hh = 25.0
hm = 8e-7

air = pd.read_excel('附件1.xlsx')
fTa = interp1d(air['时间'], air['温度'], kind='linear', fill_value='extrapolate')
fCa = interp1d(air['时间'], air['水分浓度'], kind='linear', fill_value='extrapolate')

nStep = int(te / dt)
tt = np.arange(0, nStep + 1) * dt
TaAll = fTa(tt)
CaAll = fCa(tt)

def updateT(T, Ta):
    a = np.zeros(Ng + 1); b = np.zeros(Ng + 1)
    c = np.zeros(Ng + 1); d = np.zeros(Ng + 1)
    coef = 4 * kk * dt / (rho * cp * dr * dr)
    b[0] = 1 + coef; c[0] = -coef; d[0] = T[0]
    for i in range(1, Ng):
        A = kk * dt / (rho * cp * dr * dr)
        a[i] = -A * (i - 0.5) / i
        c[i] = -A * (i + 0.5) / i
        b[i] = 1 - a[i] - c[i]; d[i] = T[i]
    a[Ng] = -kk / dr; b[Ng] = kk / dr + hh; d[Ng] = hh * Ta
    return tdma(a, b, c, d)

def updateC(C, Ca):
    a = np.zeros(Ng + 1); b = np.zeros(Ng + 1)
    c = np.zeros(Ng + 1); d = np.zeros(Ng + 1)
    D = 7e-9 * np.exp(-0.89 / np.maximum(C, 1e-6))
    coef = 4 * D[0] * dt / (dr * dr)
    b[0] = 1 + coef; c[0] = -coef; d[0] = C[0]
    for i in range(1, Ng):
        Dp = 0.5 * (D[i] + D[i + 1]); Dm = 0.5 * (D[i - 1] + D[i])
        Ap = Dp * dt / (dr * dr); Am = Dm * dt / (dr * dr)
        a[i] = -Am * (i - 0.5) / i; c[i] = -Ap * (i + 0.5) / i
        b[i] = 1 - a[i] - c[i]; d[i] = C[i]
    a[Ng] = -D[Ng] / dr; b[Ng] = D[Ng] / dr + hm; d[Ng] = hm * Ca
    return tdma(a, b, c, d)

def solveT(N):
    dR = R / N
    T = np.full(N + 1, T0)
    for n in range(nStep):
        a = np.zeros(N + 1); b = np.zeros(N + 1)
        c = np.zeros(N + 1); d = np.zeros(N + 1)
        coef = 4 * kk * dt / (rho * cp * dR * dR)
        b[0] = 1 + coef; c[0] = -coef; d[0] = T[0]
        for i in range(1, N):
            A = kk * dt / (rho * cp * dR * dR)
            a[i] = -A * (i - 0.5) / i; c[i] = -A * (i + 0.5) / i
            b[i] = 1 - a[i] - c[i]; d[i] = T[i]
        a[N] = -kk / dR; b[N] = kk / dR + hh; d[N] = hh * TaAll[n + 1]
        T = tdma(a, b, c, d)
    rp = np.arange(0, 2.001, 0.1) / 100
    return np.interp(rp, np.linspace(0, R, N + 1), T)

print('网格无关性验证...')
T50 = solveT(50); T100 = solveT(100); T200 = solveT(200)
print('1800s 温度')
for i, r in enumerate([0, 0.5, 1, 1.5, 2]):
    j = i * 5
    print(f'{r:<8} {T50[j]:<8.4f} {T100[j]:<8.4f} {T200[j]:<8.4f} {abs(T100[j]-T200[j]):.4f}')

T = np.full(Ng + 1, T0); C = np.full(Ng + 1, C0)
Ts = np.zeros((nStep + 1, Ng + 1)); Cs = np.zeros((nStep + 1, Ng + 1))
Ts[0] = T; Cs[0] = C
for n in range(nStep):
    T = updateT(T, TaAll[n + 1]); C = updateC(C, CaAll[n + 1])
    Ts[n + 1] = T; Cs[n + 1] = C
    if n % 600 == 0: print('  n =', n)

rIn = np.linspace(0, R, Ng + 1); rOut = np.arange(0, 2.001, 0.1); rOutM = rOut / 100.0
To = np.zeros((nStep + 1, len(rOut))); Co = np.zeros((nStep + 1, len(rOut)))
for n in range(nStep + 1):
    To[n, :] = np.interp(rOutM, rIn, Ts[n, :]); Co[n, :] = np.interp(rOutM, rIn, Cs[n, :])

colNames = [f'{r:.1f}' for r in rOut]
dfT = pd.DataFrame(To, columns=colNames); dfT.insert(0, '时间', tt)
dfC = pd.DataFrame(Co, columns=colNames); dfC.insert(0, '时间', tt)
with pd.ExcelWriter('result1.xlsx') as writer:
    dfT.to_excel(writer, sheet_name='温度', index=False, float_format='%.4f')
    dfC.to_excel(writer, sheet_name='水分浓度', index=False, float_format='%.4f')

tShow = [100, 300, 600, 900, 1200, 1500, 1800]; rIdx = [0, 5, 10, 15, 20]; rCm = [0, 0.5, 1, 1.5, 2]
rowsT = []; rowsC = []
for tt_ in tShow:
    rowsT.append([To[tt_, i] for i in rIdx]); rowsC.append([Co[tt_, i] for i in rIdx])
dfT1 = pd.DataFrame(np.array(rowsT).round(4), index=tShow, columns=rCm); dfT1.index.name = '时间/s'
dfC1 = pd.DataFrame(np.array(rowsC).round(4), index=tShow, columns=rCm); dfC1.index.name = '时间/s'
dfT1.to_excel('表1_温度_30min.xlsx'); dfC1.to_excel('表2_水分浓度_30min.xlsx')
print('solve1 全部完成')