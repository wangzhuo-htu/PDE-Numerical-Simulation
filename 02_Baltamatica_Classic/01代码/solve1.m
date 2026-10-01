function solve1()
% 问题一：预热平衡阶段温度与水分浓度变化模型（向量化提速版）

    %% 参数设置
    R = 0.02; Ng = 100; dr = R / Ng;
    dt = 1.0; te = 1800;
    T0 = 28.0; C0 = 2.55;
    rho = 820.0; cp = 2600.0; kk = 0.36;
    hh = 25.0; hm = 8e-7;

    %% 读取附件1数据
    airData = readmatrix('附件1.xlsx');
    tAir = airData(:, 1);
    TaAir = airData(:, 2);
    CaAir = airData(:, 3);

    %% 时间步
    nStep = floor(te / dt);
    tt = (0:nStep)' * dt;

    %% 线性插值得到逐秒边界条件
    TaAll = lin_interp(tAir, TaAir, tt);
    CaAll = lin_interp(tAir, CaAir, tt);

    %% 初始化场
    T = T0 * ones(Ng+1, 1);
    C = C0 * ones(Ng+1, 1);
    Ts = zeros(nStep+1, Ng+1);
    Cs = zeros(nStep+1, Ng+1);
    Ts(1, :) = T';
    Cs(1, :) = C';

    %% 时间推进
    for n = 1:nStep
        Ta = TaAll(n+1);
        Ca = CaAll(n+1);
        T = updateT(T, Ta, Ng, dr, dt, rho, cp, kk, hh);
        C = updateC(C, Ca, Ng, dr, dt, hm);
        Ts(n+1, :) = T';
        Cs(n+1, :) = C';
        if mod(n, 600) == 0
            fprintf('  n = %d\n', n);
        end
    end

    %% 插值到输出网格
    rIn = linspace(0, R, Ng+1)';
    rOut = (0:0.1:2.0)';
    rOutM = rOut / 100.0;
    nOut = length(rOut);

    To = zeros(nStep+1, nOut);
    Co = zeros(nStep+1, nOut);
    for n = 1:nStep+1
        To(n, :) = lin_interp(rIn, Ts(n, :)', rOutM)';
        Co(n, :) = lin_interp(rIn, Cs(n, :)', rOutM)';
    end

    %% 保存结果
    writematrix([tt, To], 'result1.xlsx', 'Sheet', '温度');
    writematrix([tt, Co], 'result1.xlsx', 'Sheet', '水分浓度');

    %% 输出表1、表2
    tShow = [100, 300, 600, 900, 1200, 1500, 1800];
    rIdx = [1, 6, 11, 16, 21];
    rowsT = zeros(length(tShow), length(rIdx));
    rowsC = zeros(length(tShow), length(rIdx));
    for i = 1:length(tShow)
        idx = tShow(i) + 1;
        rowsT(i, :) = To(idx, rIdx);
        rowsC(i, :) = Co(idx, rIdx);
    end
    writematrix(rowsT, '表1_温度_30min.xlsx');
    writematrix(rowsC, '表2_水分浓度_30min.xlsx');

    fprintf('solve1 全部完成\n');
end

%% ================= 以下为向量化子函数 =================

function T = updateT(T, Ta, Ng, dr, dt, rho, cp, kk, hh)
    a = zeros(Ng+1, 1); b = zeros(Ng+1, 1);
    c = zeros(Ng+1, 1); d = zeros(Ng+1, 1);
    
    % 中心节点
    coef = 4 * kk * dt / (rho * cp * dr * dr);
    b(1) = 1 + coef; c(1) = -coef; d(1) = T(1);
    
    % 内部节点（向量化）
    i_vec = (2:Ng)';
    A = kk * dt / (rho * cp * dr * dr);
    a(2:Ng) = -A * (i_vec - 1.5) ./ (i_vec - 1);
    c(2:Ng) = -A * (i_vec - 0.5) ./ (i_vec - 1);
    b(2:Ng) = 1 - a(2:Ng) - c(2:Ng);
    d(2:Ng) = T(2:Ng);
    
    % 表面节点
    a(Ng+1) = -kk / dr;
    b(Ng+1) = kk / dr + hh;
    d(Ng+1) = hh * Ta;
    
    T = tdma(a, b, c, d);
end

function C = updateC(C, Ca, Ng, dr, dt, hm)
    a = zeros(Ng+1, 1); b = zeros(Ng+1, 1);
    c = zeros(Ng+1, 1); d = zeros(Ng+1, 1);
    
    D = 7e-9 * exp(-0.89 ./ max(C, 1e-6));
    
    % 中心节点
    coef = 4 * D(1) * dt / (dr * dr);
    b(1) = 1 + coef; c(1) = -coef; d(1) = C(1);
    
    % 内部节点（向量化）
    i_vec = (2:Ng)';
    Dp = 0.5 * (D(2:Ng) + D(3:Ng+1));
    Dm = 0.5 * (D(1:Ng-1) + D(2:Ng));
    Ap = Dp * dt / (dr * dr);
    Am = Dm * dt / (dr * dr);
    a(2:Ng) = -Am .* (i_vec - 1.5) ./ (i_vec - 1);
    c(2:Ng) = -Ap .* (i_vec - 0.5) ./ (i_vec - 1);
    b(2:Ng) = 1 - a(2:Ng) - c(2:Ng);
    d(2:Ng) = C(2:Ng);
    
    % 表面节点
    a(Ng+1) = -D(Ng+1) / dr;
    b(Ng+1) = D(Ng+1) / dr + hm;
    d(Ng+1) = hm * Ca;
    
    C = tdma(a, b, c, d);
end

function x = tdma(a, b, c, d)
    n = length(d);
    cp = zeros(n, 1); dp = zeros(n, 1); x = zeros(n, 1);
    cp(1) = c(1) / b(1);
    dp(1) = d(1) / b(1);
    for i = 2:n
        m = b(i) - a(i) * cp(i-1);
        if i < n
            cp(i) = c(i) / m;
        else
            cp(i) = 0.0;
        end
        dp(i) = (d(i) - a(i) * dp(i-1)) / m;
    end
    x(n) = dp(n);
    for i = n-1:-1:1
        x(i) = dp(i) - cp(i) * x(i+1);
    end
end

function yi = lin_interp(x, y, xi)
    x = x(:); y = y(:); xi = xi(:);
    n = length(x);
    yi = zeros(size(xi));
    for k = 1:length(xi)
        xk = xi(k);
        if xk <= x(1)
            yi(k) = y(1);
        elseif xk >= x(n)
            yi(k) = y(n);
        else
            lo = 1; hi = n;
            while hi - lo > 1
                mid = floor((lo + hi) / 2);
                if x(mid) <= xk
                    lo = mid;
                else
                    hi = mid;
                end
            end
            t = (xk - x(lo)) / (x(hi) - x(lo));
            yi(k) = y(lo) + t * (y(hi) - y(lo));
        end
    end
end