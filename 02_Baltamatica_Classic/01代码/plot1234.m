function plot1234()
% 终极防错版绘图脚本
% 特点：首次读取后已生成 plot_cache.mat，后续秒出图；自动处理数据行数不足的情况。

    clc;
    fprintf('==================================================\n');
    
    %% ================= 智能读取数据 =================
    if exist('plot_cache.mat', 'file')
        fprintf('检测到历史数据缓存，正在秒速加载...\n');
        load('plot_cache.mat');
    else
        fprintf('未找到缓存，请先确保已经运行过 solve1~4 生成了 result1~4.xlsx！\n');
        return;
    end
    fprintf('==================================================\n');

    %% ==================== 绘制全部图表 ====================
    rOut = 0:0.1:2.0;
    colors = lines(7); 

    % ----- 图2：问题一 温度 -----
    figure('Name', '图2 问题一温度沿半径分布', 'Position', [100, 100, 550, 500]);
    hold on; box on;
    tShow1 = [100, 300, 600, 900, 1200, 1500, 1800];
    maxRows1 = size(To, 1); % 获取实际的最大行数
    for i = 1:length(tShow1)
        idx = min(tShow1(i) + 1, maxRows1); % 如果超出，就用最后一行
        plot(rOut, To(idx, :), 'o-', 'Color', colors(i,:), 'LineWidth', 1.5, 'MarkerSize', 4, ...
             'DisplayName', ['t=' num2str(tShow1(i)) 's']);
    end
    xlabel('到药材中心距离 / cm'); ylabel('温度 / ℃'); title('不同时刻温度沿半径分布');
    legend('Location', 'southeast'); grid on; axis square;

    % ----- 图4：问题一 水分 -----
    figure('Name', '图4 问题一水分沿半径分布', 'Position', [200, 100, 550, 500]);
    hold on; box on;
    maxRowsCo = size(Co, 1);
    for i = 1:length(tShow1)
        idx = min(tShow1(i) + 1, maxRowsCo);
        plot(rOut, Co(idx, :), 's-', 'Color', colors(i,:), 'LineWidth', 1.5, 'MarkerSize', 4, ...
             'DisplayName', ['t=' num2str(tShow1(i)) 's']);
    end
    xlabel('到药材中心距离 / cm'); ylabel('水分浓度 / (kg/kg)'); title('不同时刻水分浓度沿半径分布');
    legend('Location', 'northeast'); grid on; axis square;

    % ----- 图7-8：问题二 温度/水分分布 -----
    tShow2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    maxRowsT2 = size(T2_data, 1);
    maxRowsC2 = size(C2_data, 1);
    figure('Name', '图7-8 问题二温度/水分沿半径分布', 'Position', [300, 50, 1000, 480]);
    subplot(1,2,1); hold on; box on;
    for i = 1:length(tShow2)
        idx = min(round(tShow2(i) * 3600) + 1, maxRowsT2);
        plot(rOut, T2_data(idx, :), 'o-', 'Color', colors(i,:), 'LineWidth', 1.2, 'DisplayName', ['t=' num2str(tShow2(i)) 'h']);
    end
    xlabel('到药材中心距离 / cm'); ylabel('温度 / ℃'); title('不同时刻温度沿半径分布'); legend('Location', 'southeast'); grid on; axis square;
    
    subplot(1,2,2); hold on; box on;
    for i = 1:length(tShow2)
        idx = min(round(tShow2(i) * 3600) + 1, maxRowsC2);
        plot(rOut, C2_data(idx, :), 's-', 'Color', colors(i,:), 'LineWidth', 1.2, 'DisplayName', ['t=' num2str(tShow2(i)) 'h']);
    end
    xlabel('到药材中心距离 / cm'); ylabel('水分浓度 / (kg/kg)'); title('不同时刻水分浓度沿半径分布'); legend('Location', 'northeast'); grid on; axis square;

    % ----- 图9：问题二 三维曲面 -----
    figure('Name', '图9 问题二三维曲面分布', 'Position', [200, 50, 1000, 480]);
    [R_mesh, T_mesh] = meshgrid(rOut, t2);
    subplot(1,2,1); 
    surf(R_mesh, T_mesh, T2_data, 'EdgeColor', 'interp'); shading interp; colormap(jet);
    xlabel('半径 / cm'); ylabel('时间 / h'); zlabel('温度 / ℃'); title('(a) 温度三维分布'); colorbar; view(-45, 30); 
    subplot(1,2,2); 
    surf(R_mesh, T_mesh, C2_data, 'EdgeColor', 'interp'); shading interp; colormap(jet);
    xlabel('半径 / cm'); ylabel('时间 / h'); zlabel('水分浓度 / (kg/kg)'); title('(b) 水分浓度三维分布'); colorbar; view(-45, 30); 

    % ----- 图10：问题二 时空热力图 -----
    figure('Name', '图10 问题二温度/水分热力图', 'Position', [400, 50, 1000, 480]);
    subplot(1,2,1); contourf(t2, rOut, T2_data', 40); colormap(jet);
    xlabel('时间 / h'); ylabel('到药材中心距离 / cm'); title('(a) 温度时空分布'); colorbar;
    subplot(1,2,2); contourf(t2, rOut, C2_data', 40); colormap(jet);
    xlabel('时间 / h'); ylabel('到药材中心距离 / cm'); title('(b) 水分浓度时空分布'); colorbar;

    % ----- 图11：问题三 水分随时间变化 -----
    figure('Name', '图11 问题三水分随时间变化', 'Position', [500, 100, 550, 500]);
    plot(t3, C3_data(:,1), 'LineWidth', 2, 'DisplayName', '中心 r=0 cm'); hold on;
    plot(t3, C3_data(:,end), 'LineWidth', 2, 'DisplayName', '表面 r=2 cm');
    plot([t3(1), t3(end)], [0.15, 0.15], 'k--', 'LineWidth', 1.5, 'DisplayName', '干燥阈值 0.15');
    xlabel('时间 / h'); ylabel('水分浓度 / (kg/kg)'); title('不同位置水分浓度随时间变化');
    legend('Location', 'northeast'); grid on; axis square;

    % ----- 图14：问题四 消融对比 -----
    figure('Name', '图14 尺寸收缩消融对比', 'Position', [600, 100, 550, 500]);
    bar([1, 2], [56.63, 22.52], 0.5, 'FaceColor', [0.2 0.4 0.6]);
    set(gca, 'XTickLabel', {'对照模型 (固定2cm)', '完整模型 (动态收缩)'});
    ylabel('烘干时间 / h'); title('尺寸收缩对烘干时间的消融对比');
    text(1, 56.63, '56.63 h', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
    text(2, 22.52, '22.52 h', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
    grid on; axis square;

    % ----- 图15：问题四 收缩条件下水分随时间变化 -----
    figure('Name', '图15 问题四收缩条件下水分变化', 'Position', [700, 100, 550, 500]);
    plot(t4, C4_data(:,1), 'LineWidth', 2, 'DisplayName', '中心 r=0 cm'); hold on;
    plot(t4, C4_surf, 'r--', 'LineWidth', 2, 'DisplayName', '药材表面');
    plot([t4(1), t4(end)], [0.15, 0.15], 'k--', 'LineWidth', 1.5, 'DisplayName', '干燥阈值 0.15');
    xlabel('时间 / h'); ylabel('水分浓度 / (kg/kg)'); title('收缩条件下不同位置水分随时间变化');
    legend('Location', 'northeast'); grid on; axis square;

    fprintf('全部图表绘制完毕！可以开始录屏了。\n');
end