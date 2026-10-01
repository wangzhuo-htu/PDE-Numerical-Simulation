function solve3()
% 问题三：全域最大水分浓度判据，确定烘干总时长（矩阵左除提速版）
% 终止条件：max(C) < 0.15 kg/kg

    %% 参数设置
    R = 0.02; Ng = 100; dr = R / Ng;
    dt = 60.0; tmax = 300000;
    T0 = 28.0; C0 = 2.55; Cg = 0.15;
    hh = 25.0; hm = 8e-7;

    %% 读取边界条件
    airData = readmatrix('附件1.xlsx');
    tAir = airData(:, 1);
    TaAir = airData(:, 2);
    CaAir = airData(:, 3);

    nStep = floor(tmax / dt);

    %% 初始化
    T = T0 * ones(Ng+1, 1);
    C = C0 * ones(Ng+1, 1);
    Ts = T'; Cs = C'; tt = 0;
    tEnd = [];

    %% 时间推进
    for n = 1:nStep
        tNow = n * dt;
        
        % 14400s (4h) 后使用恒温恒湿外推
        if tNow <= 14400
            Ta = lin_interp(tAir, TaAir, tNow);
            Ca = lin_interp(tAir, CaAir, tNow);
        else
            Ta = 50.165; Ca = 0.04986;
        end

        % 更新温度和水分（变物性）
        T = updateT_var(T, C, Ta, Ng, dr, dt, hh);
        C = updateC_var(C, T, Ca, Ng, dr, dt, hm);
        
        % 物理限幅：表面水分不低于环境平衡浓度
        C(end) = max(C(end), 0.05);

        % 记录数据
        tt = [tt; tNow];
        Ts = [Ts; T'];
        Cs = [Cs; C'];

        % 打印进度（每模拟1小时打印一次）
        if mod(n, 60) == 0
            fprintf('  t = %.2f h, maxC = %.4f\n', tNow/3600, max(C));
        end

        % 全域终点判据：所有径向节点的水分浓度均低于阈值
        if max(C) < Cg
            tEnd = tNow;
            fprintf('  达标时间 t = %d s = %.2f h\n', tNow, tNow/3600);
            break;
        end
    end

    %% 插值输出（将内部网格插值到 0.1cm 的间隔）
    rIn = linspace(0, R, Ng+1)';
    rOut = (0:0.1:2.0)';
    rOutM = rOut / 100.0;
    nOut = length(rOut);
    nTime = length(tt);

    Co = zeros(nTime, nOut);
    for n = 1:nTime
        Co(n, :) = lin_interp(rIn, Cs(n, :)', rOutM)';
    end

    %% 保存 result3.xlsx
    writematrix([tt, Co], 'result3.xlsx', 'Sheet', '水分浓度');

    %% 保存表5（每隔6h及最终时刻的水分浓度）
    if ~isempty(tEnd)
        tShow = [6:6:floor(tEnd/3600), tEnd/3600];
        rIdx = [1, 6, 11, 16, 21];  % 对应 0, 0.5, 1.0, 1.5, 2.0 cm
        rows = zeros(length(tShow), length(rIdx));
        for i = 1:length(tShow)
            idx = min(round(tShow(i)*3600/dt) + 1, nTime);
            rows(i, :) = Co(idx, rIdx);
        end
        writematrix(rows, '表5_水分浓度_烘干过程.xlsx');
    end

    fprintf('solve3 完成\n');
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
    A = zeros(Ng+1, Ng+1);
    d = zeros(Ng+1, 1);

    % 中心节点 (i=1)
    coef = 4 * kk(1) * dt / (rho(1) * cp(1) * dr * dr);
    A(1,1) = 1 + coef; A(1,2) = -coef;
    d(1) = T(1);

    % 内部节点 (i=2..Ng)
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

    % 表面节点 (i=Ng+1)
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