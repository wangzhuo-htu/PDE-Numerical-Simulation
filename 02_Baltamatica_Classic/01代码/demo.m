function demo()
% 极简演示脚本：用于麒麟虚拟机快速展示传热传质模型运行
% 参数极度精简，计算量极小，绝不卡顿

    %% 1. 极简参数设置
    R = 0.02;          % 半径 m
    Ng = 10;           % 网格数改为 10
    dr = R / Ng;       % 空间步长
    dt = 60;           % 时间步长改为 60 秒
    te = 300;          % 总时长改为 300 秒
    T0 = 28.0;         % 初始温度
    C0 = 2.55;         % 初始水分浓度
    rho = 820.0;       % 密度
    cp = 2600.0;       % 比热容
    kk = 0.36;         % 导热系数
    hh = 25.0;         % 对流换热系数
    hm = 8e-7;         % 对流传质系数

    % 使用常数边界条件，避免读取Excel造成虚拟机卡顿
    Ta = 41.5;         % 烘房空气温度
    Ca = 0.05;         % 烘房空气水分浓度

    nStep = floor(te / dt); % 仅 5 个时间步

    %% 2. 初始化场
    T = T0 * ones(Ng+1, 1);
    C = C0 * ones(Ng+1, 1);

    %% 3. 时间推进（仅 5 次循环）
    for n = 1:nStep
        T = updateT(T, Ta, Ng, dr, dt, rho, cp, kk, hh);
        C = updateC(C, Ca, Ng, dr, dt, hm);
    end

    %% 4. 打印成功信息
    fprintf('麒麟系统演示运行成功！\n');

    %% 5. 绘制温度分布曲线图
    rOut = linspace(0, R, Ng+1);
    figure('Name', '极简演示 - 温度分布', 'Position', [300, 300, 500, 400]);
    plot(rOut, T, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#2E5A88');
    xlabel('径向位置 / m'); 
    ylabel('温度 / ℃'); 
    title('麒麟系统极简演示：温度径向分布');
    grid on; axis square;
end

%% ================= 子函数 =================

function T = updateT(T, Ta, Ng, dr, dt, rho, cp, kk, hh)
    A = zeros(Ng+1, Ng+1);
    d = zeros(Ng+1, 1);
    
    % 中心节点
    coef = 4 * kk * dt / (rho * cp * dr * dr);
    A(1,1) = 1 + coef; A(1,2) = -coef;
    d(1) = T(1);
    
    % 内部节点
    for i = 2:Ng
        a_i = -kk * dt / (rho * cp * dr * dr) * (i - 1.5) / (i - 1);
        c_i = -kk * dt / (rho * cp * dr * dr) * (i - 0.5) / (i - 1);
        b_i = 1 - a_i - c_i;
        A(i, i-1) = a_i; A(i, i) = b_i; A(i, i+1) = c_i;
        d(i) = T(i);
    end
    
    % 表面节点（第三类边界条件）
    A(Ng+1, Ng) = -kk / dr;
    A(Ng+1, Ng+1) = kk / dr + hh;
    d(Ng+1) = hh * Ta;
    
    T = A \ d;
end

function C = updateC(C, Ca, Ng, dr, dt, hm)
    A = zeros(Ng+1, Ng+1);
    d = zeros(Ng+1, 1);
    D = 7e-9 * exp(-0.89 ./ max(C, 1e-6));
    
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
        A(i, i-1) = a_i; A(i, i) = b_i; A(i, i+1) = c_i;
        d(i) = C(i);
    end
    
    % 表面节点（第三类传质边界条件）
    A(Ng+1, Ng) = -D(Ng+1) / dr;
    A(Ng+1, Ng+1) = D(Ng+1) / dr + hm;
    d(Ng+1) = hm * Ca;
    
    C = A \ d;
end