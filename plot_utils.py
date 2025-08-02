import matplotlib.pyplot as plt
import numpy as np
import os

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

def plot_cumulative_curves(eta_list, s1_total, TW_total, s1_0=None, TW_0=None):
    """
    绘制累积曲线图，包括作物面积和总用水量
    
    参数:
    eta_list: 效率提升序列
    s1_total: 各作物总面积字典
    TW_total: 各作物总用水量字典
    s1_0: 初始面积字典（可选）
    TW_0: 初始用水量字典（可选）
    """
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # 第一个子图 - 作物面积
    ax1.fill_between(eta_list, 0, s1_total['Wheat'], label='wheat', color='gold', alpha=0.8)
    ax1.fill_between(eta_list, s1_total['Wheat'], 
                     [s1_total['Wheat'][i] + s1_total['Maize'][i] for i in range(len(s1_total['Wheat']))], 
                     label='maize', color='green', alpha=0.8)
    ax1.fill_between(eta_list, 
                     [s1_total['Wheat'][i] + s1_total['Maize'][i] for i in range(len(s1_total['Wheat']))], 
                     [s1_total['Wheat'][i] + s1_total['Maize'][i] + s1_total['Rice'][i] for i in range(len(s1_total['Wheat']))], 
                     label='rice', color='#34A5DA', alpha=0.8)
    
    if s1_0:
        ax1.scatter(eta_list[0], s1_0['Wheat'], color='red', alpha=1)
        ax1.scatter(eta_list[0], s1_0['Wheat']+s1_0['Maize'], color='red', alpha=1)
        ax1.scatter(eta_list[0], s1_0['Wheat']+s1_0['Maize']+s1_0['Rice'], color='red', alpha=1)
    
    ax1.set_xlabel('Irrigation efficiency (η)', fontsize=14)
    ax1.set_ylabel('Cumulative crop planting area (ha)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', frameon=False)
    
    # 第二个子图 - 总用水量
    ax2.fill_between(eta_list, 0, TW_total['Wheat'], label='wheat', color='gold', alpha=0.8)
    ax2.fill_between(eta_list, TW_total['Wheat'], 
                     [TW_total['Wheat'][i] + TW_total['Maize'][i] for i in range(len(TW_total['Wheat']))], 
                     label='maize', color='green', alpha=0.8)
    ax2.fill_between(eta_list, 
                     [TW_total['Wheat'][i] + TW_total['Maize'][i] for i in range(len(TW_total['Wheat']))], 
                     [TW_total['Wheat'][i] + TW_total['Maize'][i] + TW_total['Rice'][i] for i in range(len(TW_total['Wheat']))], 
                     label='rice', color='#34A5DA', alpha=0.8)
    
    if TW_0:
        ax2.scatter(eta_list[0], TW_0['Wheat'], color='red', alpha=1)
        ax2.scatter(eta_list[0], TW_0['Wheat']+TW_0['Maize'], color='red', alpha=1)
        ax2.scatter(eta_list[0], TW_0['Wheat']+TW_0['Maize']+TW_0['Rice'], color='red', alpha=1)
    
    ax2.set_xlabel('Irrigation efficiency (η)', fontsize=14)
    ax2.set_ylabel('Cumulative water use (m³)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.show()



def plot_w_s_star_ranges(eta_list, w_star_range, s1_total, crop_types):
    """
    绘制w_star范围图，显示每个作物的最优灌溉量范围
    
    参数:
    eta_list: 效率提升序列
    w_star_range: 各作物w_star范围字典
    s1_total: 各作物总面积字典
    crop_types: 作物类型列表
    """
    plt.close('all')  # 清除所有已存在的图
    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(20, 6))
    
    for crop, ax, label in zip(crop_types, [ax3, ax4, ax5], ['(a)', '(b)', '(c)']):
        # 创建双坐标轴
        ax2 = ax.twinx()
        
        w_star_median = []
        w_star_min = []
        w_star_max = []
        
        for i in range(len(eta_list)):
            w_star_median.append(w_star_range[crop][i][1]/10)
            w_star_min.append(w_star_range[crop][i][0]/10)
            w_star_max.append(w_star_range[crop][i][2]/10)
        
        # 在左侧坐标轴绘制w_star范围
        ax.plot(eta_list, w_star_median, color='black', alpha=0.8)
        ax.plot(eta_list, w_star_min, color='black', alpha=0.8, linestyle='--')
        ax.plot(eta_list, w_star_max, color='black', alpha=0.8, linestyle='--')
        ax.fill_between(eta_list, w_star_min, w_star_max, 
                     label='Irrigation amount', color='blue', alpha=0.2)
        
        # 对list中的每个元素除以100
        s1_list = [x/100 for x in s1_total[crop]]
        ax2.plot(eta_list, s1_list, color='red', alpha=0.6, linewidth=2, label='Total irrigated area')
        
        # 设置左侧坐标轴
        ax.set_xlabel('Irrigation efficiency ($\eta$)', fontsize=16)
        ax.set_ylabel('Irrigation amount (mm)', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f'{label} {crop}', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(300, 1700)
        
        # 设置右侧坐标轴
        ax2.set_ylabel('Irrigated area (km$^2$)', fontsize=16)
        ax2.tick_params(axis='y', which='major', labelsize=16)
        ax2.set_ylim(100, 1500)
        
        # 合并两个坐标轴的图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False, fontsize=16)
    
    plt.tight_layout()
    
    # 先保存图片
    save_dir = r'F:\AquaCrop-Paradox\Figures'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'w_s_star_ranges.tif'), dpi=900, format='tif', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'w_s_star_ranges.pdf'), dpi=900, format='pdf', bbox_inches='tight')
    
    # 最后显示图片
    plt.show()

def annotate_max_point(ax, x_data, y_data, offset_x=0.01, offset_y=0, point_size=50, 
                      line_width=1, font_size=16, color='black'):
    """
    在曲线上标注最大值点
    
    参数:
    ax: matplotlib轴对象
    x_data: x轴数据
    y_data: y轴数据
    offset_x: x轴标注的水平偏移量
    offset_y: y轴标注的垂直偏移量
    point_size: 点的大小
    line_width: 线的宽度
    font_size: 字体大小
    color: 点和线的颜色
    """
    # 找到最大值点
    max_idx = np.argmax(y_data)
    max_x = x_data[max_idx]
    max_y = y_data[max_idx]
    
    # 绘制最高点
    ax.scatter(max_x, max_y, color='red', s=point_size, zorder=8)
    
    # 绘制垂直线
    ax.plot([max_x, max_x], [0, max_y], color=color, linestyle='--', linewidth=line_width)
    
    # 标注x值
    ax.text(max_x + offset_x, offset_y, f'$\eta$ = {max_x:.2f}', 
           transform=ax.transData, va='bottom', ha='left',
           fontsize=font_size)
    
    return max_x, max_y

def plot_water_use(eta_list, TW_total, crop_types):
    """
    绘制用水量图，包括三种作物的单独用水量和总用水量的累积曲线
    
    参数:
    eta_list: 效率提升序列
    TW_total: 各作物总用水量字典
    crop_types: 作物类型列表
    """
    plt.close('all')  # 清除所有已存在的图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    
    # 定义颜色
    maize_color = [0, 0.5, 0]  # 深绿色
    wheat_color = [0.6, 0.3, 0]  # 棕色
    rice_color = [0, 0.4, 0.8]  # 蓝色
    
    # 计算总用水量
    total_water = [sum(TW_total[crop][i] for crop in crop_types) for i in range(len(eta_list))]
    
    # 设置科学计数法格式
    def format_sci(x, pos):
        if x == 0:
            return '0'
        return f'{x/1e8:.1f}'
    
    # 绘制三个作物的单独用水量
    for i, (crop, ax, label, color) in enumerate(zip(crop_types, [ax1, ax2, ax3], ['(a)', '(b)', '(c)'], 
                                    [rice_color, wheat_color, maize_color])):
        # 绘制曲线
        ax.plot(eta_list, TW_total[crop], color='black', alpha=0.8, linewidth=2)
        # 填充曲线下方区域
        ax.fill_between(eta_list, 0, TW_total[crop], color=color, alpha=0.3)
        
        # 如果是玉米，标注最大值点
        if crop == 'Maize':
            annotate_max_point(ax, eta_list, TW_total[crop])
        
        ax.set_xlabel('Irrigation efficiency ($\eta$)', fontsize=16)
        ax.set_ylabel('Total water use (m$^3$)', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f'{label} {crop}', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # 设置y轴格式
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_sci))
        
        # 设置y轴范围
        if i < 2:  # 前两个子图
            ax.set_ylim(0, 4.3e8)
        else:  # 第三个子图
            ax.set_ylim(0, 8.3e8)
            
        # 在左上角添加10^8标注
        ax.text(-0.02, 1.05, '×10$^8$', transform=ax.transAxes, 
                fontsize=16, verticalalignment='top')
    
    # 绘制总用水量累积曲线（从下到上：玉米、小麦、水稻）
    # 计算分界线
    maize_wheat_boundary = [TW_total['Maize'][i] for i in range(len(eta_list))]
    wheat_rice_boundary = [TW_total['Maize'][i] + TW_total['Wheat'][i] for i in range(len(eta_list))]

    
    # 绘制填充区域
    ax4.fill_between(eta_list, 0, maize_wheat_boundary, label='Maize', color=maize_color, alpha=0.6)
    ax4.fill_between(eta_list, maize_wheat_boundary, wheat_rice_boundary, 
                     label='Wheat', color=wheat_color, alpha=0.6)
    ax4.fill_between(eta_list, wheat_rice_boundary, total_water, 
                     label='Rice', color=rice_color, alpha=0.6)
    
    # 绘制分界线
    ax4.plot(eta_list, maize_wheat_boundary, color='black', linewidth=1)
    ax4.plot(eta_list, wheat_rice_boundary, color='black', linewidth=1)
    ax4.plot(eta_list, total_water, color='black', linewidth=1)

    ax4.set_xlabel('Irrigation efficiency ($\eta$)', fontsize=16)
    ax4.set_ylabel('Cumulative total water use (m$^3$)', fontsize=16)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_title('(d) Irrigation district', fontsize=18, fontweight='bold')
    ax4.tick_params(axis='both', which='major', labelsize=16)
    ax4.legend(loc='upper right', frameon=False, fontsize=16)
    annotate_max_point(ax4, eta_list, total_water)
    # 设置y轴格式和范围
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(format_sci))
    y_max = max(total_water) * 1.4
    ax4.set_ylim(0, y_max)
    
    # 在左上角添加10^8标注
    ax4.text(-0.02, 1.05, '×10$^8$', transform=ax4.transAxes, 
            fontsize=16, verticalalignment='top')
    
    plt.tight_layout()
    
    # 保存图片
    save_dir = r'G:\AquaCrop-Paradox\Figures'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'total_water_use2.tif'), dpi=900, format='tif', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'total_water_use2.pdf'), dpi=900, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'total_water_use2.jpg'), dpi=900, format='jpg', bbox_inches='tight')
    plt.show()

def plot_pw_water_use_heatmap(eta_list, pw_list, all_TW_total, crop_types):
    """
    绘制不同效率和水价组合下的总用水量热力图
    
    参数:
    eta_list: 效率提升序列
    pw_list: 水价序列
    all_TW_total: 不同水价下的总用水量字典
    crop_types: 作物类型列表
    """
    plt.close('all')
    
    # 创建数据矩阵
    total_water_matrix = np.zeros((len(pw_list), len(eta_list)))
    
    # 填充数据矩阵
    for i, pw in enumerate(pw_list):
        for j, eta in enumerate(eta_list):
            # 计算该(eta, pw)组合下的总用水量
            total_water = sum(all_TW_total[pw][crop][j] for crop in crop_types)
            total_water_matrix[i, j] = total_water / 1e8  # 转换为亿立方米
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 计算刻度的位置（格子中心）
    eta_centers = (eta_list[:-1] + eta_list[1:]) / 2
    eta_centers = np.insert(eta_centers, 0, eta_list[0] - (eta_list[1] - eta_list[0])/2)
    eta_centers = np.append(eta_centers, eta_list[-1] + (eta_list[-1] - eta_list[-2])/2)
    
    pw_centers = (pw_list[:-1] + pw_list[1:]) / 2
    pw_centers = np.insert(pw_centers, 0, pw_list[0] - (pw_list[1] - pw_list[0])/2)
    pw_centers = np.append(pw_centers, pw_list[-1] + (pw_list[-1] - pw_list[-2])/2)
    
    # 绘制热力图
    im = ax.imshow(total_water_matrix, cmap='YlOrRd', aspect='auto', 
                   extent=[eta_centers[0], 1.0, pw_centers[0], pw_centers[-1]],
                   origin='lower')
    
    # 绘制等值曲线
    # 创建网格点
    eta_grid, pw_grid = np.meshgrid(eta_list, pw_list)
    # 绘制指定的等值曲线
    contour = ax.contour(eta_grid, pw_grid, total_water_matrix, 
                        levels=[8.8, 9.0, 9.2, 9.5],
                        colors='black', linewidths=1.5, alpha=1)
    # 添加等值线标签，显示为×10^8，加粗
    ax.clabel(contour, inline=True, fontsize=14, fmt=lambda x: f'{x:.1f}×10$^8$')
    
    # 设置刻度位置和标签
    # 创建每隔0.4的刻度值
    eta_ticks = np.arange(eta_list[0], 1.01, 0.04)
    ax.set_xticks(eta_ticks)
    ax.set_yticks(pw_list)
    
    # 设置刻度标签
    ax.set_xticklabels([f'{x:.2f}' for x in eta_ticks])
    ax.set_yticklabels([f'{y:.3f}' for y in pw_list])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total water use (×10$^8$ m$^3$)', fontsize=16)
    
    # 设置坐标轴标签
    ax.set_xlabel('Irrigation efficiency ($\eta$)', fontsize=16)
    ax.set_ylabel('Water price (Yuan/m$^3$)', fontsize=16)
    
    # 设置刻度
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    save_dir = r'G:\AquaCrop-Paradox\Figures'
    os.makedirs(save_dir, exist_ok=True)
    #plt.savefig(os.path.join(save_dir, 'water_use_heatmap.tif'), dpi=900, format='tif', bbox_inches='tight')
    #plt.savefig(os.path.join(save_dir, 'water_use_heatmap.pdf'), dpi=900, format='pdf', bbox_inches='tight')
    #plt.savefig(os.path.join(save_dir, 'water_use_heatmap.jpg'), dpi=900, format='jpg', bbox_inches='tight')
    plt.show()
    return total_water_matrix


def plot_pc_water_use_heatmap(eta_list, p_list1, p_list2, all_TW_total1, all_TW_total2, crop_types):
    """
    绘制不同效率和水价组合下的总用水量热力图，并排显示两个场景
    
    参数:
    eta_list: 效率提升序列
    p_list1: 第一个场景的价格序列
    p_list2: 第二个场景的价格序列
    all_TW_total1: 第一个场景下不同价格的总用水量字典
    all_TW_total2: 第二个场景下不同价格的总用水量字典
    crop_types: 作物类型列表
    """
    plt.close('all')
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 创建数据矩阵
    total_water_matrix1 = np.zeros((len(p_list1), len(eta_list)))
    total_water_matrix2 = np.zeros((len(p_list2), len(eta_list)))
    
    # 填充数据矩阵
    for i, p in enumerate(p_list1):
        for j, eta in enumerate(eta_list):
            # 计算该(eta, pw)组合下的总用水量
            total_water1 = sum(all_TW_total1[p][crop][j] for crop in crop_types)
            total_water_matrix1[i, j] = total_water1 / 1e8  # 转换为亿立方米
            
    for i, p in enumerate(p_list2):
        for j, eta in enumerate(eta_list):
            total_water2 = sum(all_TW_total2[p][crop][j] for crop in crop_types)
            total_water_matrix2[i, j] = total_water2 / 1e8  # 转换为亿立方米
    
    # 计算刻度的位置（格子中心）
    eta_centers = (eta_list[:-1] + eta_list[1:]) / 2
    eta_centers = np.insert(eta_centers, 0, eta_list[0] - (eta_list[1] - eta_list[0])/2)
    eta_centers = np.append(eta_centers, eta_list[-1] + (eta_list[-1] - eta_list[-2])/2)
    
    p_centers1 = (p_list1[:-1] + p_list1[1:]) / 2
    p_centers1 = np.insert(p_centers1, 0, p_list1[0] - (p_list1[1] - p_list1[0])/2)
    p_centers1 = np.append(p_centers1, p_list1[-1] + (p_list1[-1] - p_list1[-2])/2)
    
    p_centers2 = (p_list2[:-1] + p_list2[1:]) / 2
    p_centers2 = np.insert(p_centers2, 0, p_list2[0] - (p_list2[1] - p_list2[0])/2)
    p_centers2 = np.append(p_centers2, p_list2[-1] + (p_list2[-1] - p_list2[-2])/2)
    
    # 绘制第一个热力图
    im1 = ax1.imshow(total_water_matrix1, cmap='YlOrRd', aspect='auto', 
                    extent=[eta_centers[0], 1.0, p_centers1[0], p_centers1[-1]],
                    origin='lower')
    
    # 绘制第二个热力图
    im2 = ax2.imshow(total_water_matrix2, cmap='YlOrRd', aspect='auto', 
                    extent=[eta_centers[0], 1.0, p_centers2[0], p_centers2[-1]],
                    origin='lower')
    
    # 为两个图绘制等值曲线
    eta_grid1, p_grid1 = np.meshgrid(eta_list, p_list1)
    eta_grid2, p_grid2 = np.meshgrid(eta_list, p_list2)
    
    contour1 = ax1.contour(eta_grid1, p_grid1, total_water_matrix1, 
                          levels=[8.7635, 9.0, 9.2, 9.5],
                          colors='black', linewidths=1.5, alpha=1)
    contour2 = ax2.contour(eta_grid2, p_grid2, total_water_matrix2, 
                          levels=[8.8, 9.0, 9.2, 9.5],
                          colors='black', linewidths=1.5, alpha=1)
    
    # 添加等值线标签
    ax1.clabel(contour1, inline=True, fontsize=14, fmt=lambda x: f'{x:.1f}×10$^8$')
    ax2.clabel(contour2, inline=True, fontsize=14, fmt=lambda x: f'{x:.1f}×10$^8$')
    
    # 设置刻度位置和标签
    eta_ticks = np.arange(eta_list[0], 1.01, 0.04)
    ax1.set_xticks(eta_ticks)
    ax2.set_xticks(eta_ticks)
    ax1.set_yticks(p_list1)
    ax2.set_yticks(p_list2)
    
    ax1.set_xticklabels([f'{x:.2f}' for x in eta_ticks])
    ax2.set_xticklabels([f'{x:.2f}' for x in eta_ticks])
    ax1.set_yticklabels([f'{y:.3f}' for y in p_list1])
    ax2.set_yticklabels([f'{y:.3f}' for y in p_list2])
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar1.set_label('Total water use (×10$^8$ m$^3$)', fontsize=16)
    cbar2.set_label('Total water use (×10$^8$ m$^3$)', fontsize=16)
    
    # 设置坐标轴标签
    ax1.set_xlabel('Irrigation efficiency ($\eta$)', fontsize=16)
    ax2.set_xlabel('Irrigation efficiency ($\eta$)', fontsize=16)
    ax1.set_ylabel('Wheat price (Yuan/kg)', fontsize=16)
    ax2.set_ylabel('Maize price (Yuan/kg)', fontsize=16)
    
    # 设置标题
    ax1.set_title('(a)', fontsize=18, fontweight='bold')
    ax2.set_title('(b)', fontsize=18, fontweight='bold')
    
    # 设置刻度
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    save_dir = r'G:\AquaCrop-Paradox\Figures'
    os.makedirs(save_dir, exist_ok=True)
    #plt.savefig(os.path.join(save_dir, 'water_use_heatmap_pc2.tif'), dpi=900, format='tif', bbox_inches='tight')
    #plt.savefig(os.path.join(save_dir, 'water_use_heatmap_pc2.pdf'), dpi=900, format='pdf', bbox_inches='tight')
    #plt.savefig(os.path.join(save_dir, 'water_use_heatmap_pc2.jpg'), dpi=900, format='jpg', bbox_inches='tight')
    
    plt.show()
    
    return total_water_matrix1, total_water_matrix2


def plot_alpha_water_use_heatmap(eta_list, alpha_list, all_TW_total, crop_types):
    """
    绘制不同效率和成本组合下的总用水量热力图
    
    参数:
    eta_list: 效率提升序列
    alpha_list: 成本序列
    all_TW_total: 不同成本下的总用水量字典
    crop_types: 作物类型列表
    """
    plt.close('all')
    
    # 创建数据矩阵
    total_water_matrix = np.zeros((len(alpha_list), len(eta_list)))
    
    # 填充数据矩阵
    for i, alpha in enumerate(alpha_list):
        for j, eta in enumerate(eta_list):
            # 计算该(eta, pw)组合下的总用水量
            total_water = sum(all_TW_total[alpha][crop][j] for crop in crop_types)
            total_water_matrix[i, j] = total_water / 1e8  # 转换为亿立方米
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 计算刻度的位置（格子中心）
    eta_centers = (eta_list[:-1] + eta_list[1:]) / 2
    eta_centers = np.insert(eta_centers, 0, eta_list[0] - (eta_list[1] - eta_list[0])/2)
    eta_centers = np.append(eta_centers, eta_list[-1] + (eta_list[-1] - eta_list[-2])/2)
    
    alpha_centers = (alpha_list[:-1] + alpha_list[1:]) / 2
    alpha_centers = np.insert(alpha_centers, 0, alpha_list[0] - (alpha_list[1] - alpha_list[0])/2)
    alpha_centers = np.append(alpha_centers, alpha_list[-1] + (alpha_list[-1] - alpha_list[-2])/2)
    
    # 绘制热力图
    im = ax.imshow(total_water_matrix, cmap='YlOrRd', aspect='auto', 
                   extent=[eta_centers[0], 1.0, alpha_centers[0], alpha_centers[-1]],
                   origin='lower')
    
    # 绘制等值曲线
    # 创建网格点
    eta_grid, alpha_grid = np.meshgrid(eta_list, alpha_list)
    # 绘制指定的等值曲线
    contour = ax.contour(eta_grid, alpha_grid, total_water_matrix, 
                        levels=[8.8, 9.0, 9.2, 9.5],
                        colors='black', linewidths=1.5, alpha=1)
    # 添加等值线标签，显示为×10^8，加粗
    ax.clabel(contour, inline=True, fontsize=14, fmt=lambda x: f'{x:.1f}×10$^8$')
    
    # 设置刻度位置和标签
    # 创建每隔0.4的刻度值
    eta_ticks = np.arange(eta_list[0], 1.01, 0.04)
    ax.set_xticks(eta_ticks)
    ax.set_yticks(alpha_list)
    
    # 设置刻度标签
    ax.set_xticklabels([f'{x:.2f}' for x in eta_ticks])
    ax.set_yticklabels([f'{y:.0f}' for y in alpha_list])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total water use (×10$^8$ m$^3$)', fontsize=16)
    
    # 设置坐标轴标签
    ax.set_xlabel('Irrigation efficiency ($\eta$)', fontsize=16)
    ax.set_ylabel('Increased land cost (Δ $\mathit{α}$) (Yuan/ha)', fontsize=16)
    
    # 设置刻度
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    save_dir = r'G:\AquaCrop-Paradox\Figures'
    os.makedirs(save_dir, exist_ok=True)
    #plt.savefig(os.path.join(save_dir, 'water_use_heatmap_alpha.tif'), dpi=900, format='tif', bbox_inches='tight')
    #plt.savefig(os.path.join(save_dir, 'water_use_heatmap_alpha.pdf'), dpi=900, format='pdf', bbox_inches='tight')
    #plt.savefig(os.path.join(save_dir, 'water_use_heatmap_alpha.jpg'), dpi=900, format='jpg', bbox_inches='tight')
    plt.show()
    return total_water_matrix