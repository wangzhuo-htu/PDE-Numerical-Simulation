function solve2()
% 问题二：变物性传热传质耦合模型（矩阵左除提速版）

    %% 参数设置
    R = 0.02; Ng = 100; dr = R / Ng;
    dt = 1.0; te = 10800;
    T0 = 28.0; C0 = 2.55;
    hh = 25.0; hm = 8e-7;

    %% 读取边界条件
    airData = readmatrix('附件1.xlsx');
    tAir = airData(:, 1);
    TaAir = airData(:, 2);
    CaAir = airData(:, 3);

    nStep = floor(te / dt);
    tt = (0:nStep)' * dt;
    TaAll = lin_interp(tAir, TaAir, tt);
    CaAll = lin_interp(tAir, CaAir, tt);

    %% 初始化
    T = T0 * ones(Ng+1, 1);
    C = C0 * ones(Ng+1, 1);
    Ts = zeros(nStep+1, Ng+1);
    Cs = zeros(nStep+1, Ng+1);
    Ts(1, :) = T'; Cs(1, :) = C';

    %% 时间推进
    for n = 1:nStep
        tNow = n * dt;
        if tNow <= 14400
            Ta = TaAll(n+1);
            Ca = CaAll(n+1);
        else
            Ta = 50.165; Ca = 0.04986;
        end
        T = updateT_var(T, C, Ta, Ng, dr, dt, hh);
        C = updateC_var(C, T, Ca, Ng, dr, dt, hm);
        Ts(n+1, :) = T';
        Cs(n+1, :) = C';
        if mod(n, 1800) == 0
            fprintf('  n = %d\n', n);
        end
    end

    %% 插值输出
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

    %% 保存 result2.xlsx
    writematrix([tt, To], 'result2.xlsx', 'Sheet', '温度');
    writematrix([tt, Co], 'result2.xlsx', 'Sheet', '水分浓度');

    %% 表3、表4
    tShow = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    rIdx = [1, 6, 11, 16, 21];
    rowsT = zeros(length(tShow), length(rIdx));
    rowsC = zeros(length(tShow), length(rIdx));
    for i = 1:length(tShow)
        idx = round(tShow(i) * 3600 / dt) + 1;
        rowsT(i, :) = To(idx, rIdx);
        rowsC(i, :) = Co(idx, rIdx);
    end
    writematrix(rowsT, '表3_温度_3h.xlsx');
    writematrix(rowsC, '表4_水分浓度_3h.xlsx');

    fprintf('3h 表面温度/水分: %.4f, %.4f\n', To(end, end), Co(end, end));
    fprintf('3h 中心温度/水分: %.4f, %.4f\n', To(end, 1), Co(end, 1));
    fprintf('solve2 完成\n');
end

%% ================= 以下为子函数 =================

function [rho, cp, kk, D] = props(C, T)
    rho = 650 + 128 * C;
    cp = 1450 + 2736 * C ./ (C + 1);
    kk = 0.21 + 0.38 * C ./ (C + 1);
    TK = T + 273.15;
    D = 2.4e-3 * exp(-0.45 ./ max(C, 1e-6)) .* exp(-3850 ./ TK);
end

function T = updateT_var(T, C, Ta, Ng, dr, dt, hh)
    [rho, cp, kk, ~] = props(C, T);
    % 构造三对角矩阵 A 和右端向量 d
    A = zeros(Ng+1, Ng+1);
    d = zeros(Ng+1, 1);

    % 中心节点
    coef = 4 * kk(1) * dt / (rho(1) * cp(1) * dr * dr);
    A(1,1) = 1 + coef; A(1,2) = -coef;
    d(1) = T(1);

    % 内部节点
    for i = 2:Ng
        kp = 0.5 * (kk(i) + kk(i+1));
        km = 0.5 * (kk(i-1) + kk(i));
        Ap = kp * dt / (rho(i) * cp(i) * dr * dr);
        Am = km * dt / (rho(i) * cp(i) * dr * dr);
        a_i = -Am * (i - 1.5) / (i - 1);
        c_i = -Ap * (i - 0.5) / (i - 1);
        b_i = 1 - a_i - c_i;
        A(i, i-1) = a_i;
        A(i, i)   = b_i;
        A(i, i+1) = c_i;
        d(i) = T(i);
    end

    % 表面节点
    A(Ng+1, Ng)   = -kk(Ng+1) / dr;
    A(Ng+1, Ng+1) = kk(Ng+1) / dr + hh;
    d(Ng+1) = hh * Ta;

    T = A \ d;
end

function C = updateC_var(C, T, Ca, Ng, dr, dt, hm)
    [~, ~, ~, D] = props(C, T);
    A = zeros(Ng+1, Ng+1);
    d = zeros(Ng+1, 1);

    % 中心节点
    coef = 4 * D(1) * dt / (dr * dr);
    A(1,1) = 1 + coef; A(1,2) = -coef;
    d(1) = C(1);

    % 内部节点
    for i = 2:Ng
        Dp = 0.5 * (D(i) + D(i+1));
        Dm = 0.5 * (D(i-1) + D(i));
        Ap = Dp * dt / (dr * dr);
        Am = Dm * dt / (dr * dr);
        a_i = -Am * (i - 1.5) / (i - 1);
        c_i = -Ap * (i - 0.5) / (i - 1);
        b_i = 1 - a_i - c_i;
        A(i, i-1) = a_i;
        A(i, i)   = b_i;
        A(i, i+1) = c_i;
        d(i) = C(i);
    end

    % 表面节点
    A(Ng+1, Ng)   = -D(Ng+1) / dr;
    A(Ng+1, Ng+1) = D(Ng+1) / dr + hm;
    d(Ng+1) = hm * Ca;

    C = A \ d;
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