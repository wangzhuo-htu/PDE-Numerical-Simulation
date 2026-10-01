function solve4()
% 问题四：移动边界模型，半径随脱水收缩（矩阵左除提速版）
% 归一化坐标 ξ = r/R(t)，物性采用附录4

    %% 参数设置
    R0 = 0.02; Ng = 100; dxi = 1.0 / Ng;
    dt = 60.0; tmax = 300000;
    T0 = 28.0; C0 = 2.55; Cg = 0.15;
    hh = 25.0; hm = 8e-7;

    %% 读取边界条件（附件1）
    airData = readmatrix('附件1.xlsx');
    tAir = airData(:, 1);
    TaAir = airData(:, 2);
    CaAir = airData(:, 3);

    %% 读取半径数据（附件2）
    radData = readmatrix('附件2.xlsx');
    tRad = radData(:, 1);
    RRad = radData(:, 2) / 100.0;  % cm -> m

    nStep = floor(tmax / dt);

    %% 初始化
    T = T0 * ones(Ng+1, 1);
    C = C0 * ones(Ng+1, 1);
    Ts = T'; Cs = C'; Rs = R0; tt = 0;
    tEnd = [];

    %% 时间推进
    for n = 1:nStep
        tNow = n * dt;
        
        % 当前时刻的半径与收缩速率
        Rnow = lin_interp(tRad, RRad, tNow);
        Rnext = lin_interp(tRad, RRad, tNow + dt);
        dR = (Rnext - Rnow) / dt;

        % 边界条件
        if tNow <= 14400
            Ta = lin_interp(tAir, TaAir, tNow);
            Ca = lin_interp(tAir, CaAir, tNow);
        else
            Ta = 50.165; Ca = 0.04986;
        end

        % 更新温度和水分浓度场
        T = updateT_mov(T, C, Rnow, dR, Ta, Ng, dxi, dt, hh);
        C = updateC_mov(C, T, Rnow, dR, Ca, Ng, dxi, dt, hm);
        
        % 物理限幅：表面水分不低于环境平衡浓度
        C(end) = max(C(end), 0.05);

        % 记录数据
        tt = [tt; tNow];
        Ts = [Ts; T'];
        Cs = [Cs; C'];
        Rs = [Rs; Rnow];

        % 打印进度（每模拟1小时打印一次，让你知道没卡住）
        if mod(n, 60) == 0
            fprintf('  t = %.2f h, R = %.3f cm, maxC = %.4f\n', ...
                tNow/3600, Rnow*100, max(C));
        end

        % 全域终点判据
        if max(C) < Cg
            tEnd = tNow;
            fprintf('  达标时间 t = %d s = %.2f h\n', tNow, tNow/3600);
            break;
        end
    end

    %% 插值输出（将归一化坐标结果映射到实际空间坐标）
    rOut = (0:0.1:2.0)';
    nOut = length(rOut);
    nTime = length(tt);
    Co = zeros(nTime, nOut + 1);  % 最后一列为表面

    for n = 1:nTime
        R = Rs(n);
        for j = 1:nOut
            rM = rOut(j) / 100.0;
            if rM <= R
                x = rM / R;
                i0 = floor(x * Ng);
                i1 = min(i0 + 1, Ng);
                w = x * Ng - i0;
                Co(n, j) = (1 - w) * Cs(n, i0+1) + w * Cs(n, i1+1);
            else
                Co(n, j) = NaN; % 超出当前半径的位置设为 NaN
            end
        end
        Co(n, end) = Cs(n, end); % 当前半径的表面水分浓度
    end

    %% 保存 result4.xlsx
    writematrix([tt, Co], 'result4.xlsx', 'Sheet', '水分浓度');

    %% 保存表6（每隔6h及最终时刻的水分浓度）
    if ~isempty(tEnd)
        tShow = [6:6:floor(tEnd/3600), tEnd/3600];
        rIdx = [1, 5, 9, 13];  % 对应 0, 0.4, 0.8, 1.2 cm
        rows = zeros(length(tShow), length(rIdx) + 1);
        for i = 1:length(tShow)
            idx = min(round(tShow(i)*3600/dt) + 1, nTime);
            rows(i, 1:length(rIdx)) = Co(idx, rIdx);
            rows(i, end) = Co(idx, end);
        end
        writematrix(rows, '表6_水分浓度_收缩模型.xlsx');
    end

    fprintf('solve4 完成\n');
end

%% ================= 以下为子函数 =================

function [rho, cp, kk, D] = props4(C, T)
    % 附录4经验公式
    rho = 760 + 90 * C;
    cp = 1850 + 2150 * C ./ (C + 1);
    kk = 0.12 + 0.20 * C ./ (C + 1);
    TK = T + 273.15;
    D = 4.2e-4 * exp(-0.30 ./ max(C, 1e-6)) .* exp(-3500 ./ TK);
end

function T = updateT_mov(T, C, R, dR, Ta, Ng, dxi, dt, hh)
    [rho, cp, kk, ~] = props4(C, T);
    A = zeros(Ng+1, Ng+1);
    d = zeros(Ng+1, 1);

    % 中心节点 (i=1)
    coef = 4 * kk(1) * dt / (rho(1) * cp(1) * R^2 * dxi^2);
    A(1,1) = 1 + coef; A(1,2) = -coef;
    d(1) = T(1);

    % 内部节点 (i=2..Ng)
    for i = 2:Ng
        % 收缩对流项（显式处理，放入右端向量）
        g = (i/Ng) * dR / R * (T(i+1) - T(i-1)) / (2*dxi) * dt;
        
        kp = 0.5 * (kk(i) + kk(i+1));
        km = 0.5 * (kk(i-1) + kk(i));
        Ap = kp * dt / (rho(i) * cp(i) * R^2 * dxi^2);
        Am = km * dt / (rho(i) * cp(i) * R^2 * dxi^2);
        a_i = -Am * (i - 1.5) / (i - 1);
        c_i = -Ap * (i - 0.5) / (i - 1);
        b_i = 1 - a_i - c_i;
        
        A(i, i-1) = a_i;
        A(i, i)   = b_i;
        A(i, i+1) = c_i;
        d(i) = T(i) + g;
    end

    % 表面节点 (i=Ng+1)
    A(Ng+1, Ng)   = -kk(Ng+1) / (R * dxi);
    A(Ng+1, Ng+1) = kk(Ng+1) / (R * dxi) + hh;
    d(Ng+1) = hh * Ta;

    T = A \ d;
end

function C = updateC_mov(C, T, R, dR, Ca, Ng, dxi, dt, hm)
    [~, ~, ~, D] = props4(C, T);
    A = zeros(Ng+1, Ng+1);
    d = zeros(Ng+1, 1);

    % 中心节点 (i=1)
    coef = 4 * D(1) * dt / (R^2 * dxi^2);
    A(1,1) = 1 + coef; A(1,2) = -coef;
    d(1) = C(1);

    % 内部节点 (i=2..Ng)
    for i = 2:Ng
        % 收缩对流项（显式处理）
        g = (i/Ng) * dR / R * (C(i+1) - C(i-1)) / (2*dxi) * dt;
        
        Dp = 0.5 * (D(i) + D(i+1));
        Dm = 0.5 * (D(i-1) + D(i));
        Ap = Dp * dt / (R^2 * dxi^2);
        Am = Dm * dt / (R^2 * dxi^2);
        a_i = -Am * (i - 1.5) / (i - 1);
        c_i = -Ap * (i - 0.5) / (i - 1);
        b_i = 1 - a_i - c_i;
        
        A(i, i-1) = a_i;
        A(i, i)   = b_i;
        A(i, i+1) = c_i;
        d(i) = C(i) + g;
    end

    % 表面节点 (i=Ng+1)
    A(Ng+1, Ng)   = -D(Ng+1) / (R * dxi);
    A(Ng+1, Ng+1) = D(Ng+1) / (R * dxi) + hm;
    d(Ng+1) = hm * Ca;

    C = A \ d;
end

function yi = lin_interp(x, y, xi)
    % 手动线性插值，完全兼容北太天元
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