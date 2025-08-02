#%% 本脚本依据Sim215Agent.py，对真实模拟情景进行敏感性分析
import importlib
import Agent
importlib.reload(Agent)
from Agent import create_farmers, calibrate_alpha_beta, read_crop_params, land_expansion
import numpy as np
import plot_utils
importlib.reload(plot_utils)
from plot_utils import plot_cumulative_curves

def initialize_simulation(eta0, pw, crop_params_file, sheet_name):
    """
    初始化模拟参数和农民数据
    
    参数:
    eta0: 初始灌溉效率
    pw: 水价
    crop_params_file: 作物参数Excel文件名
    sheet_name: Excel表格名称
    """
    crop_params_list = read_crop_params(crop_params_file, sheet_name)
    crop_types = ['Rice', 'Wheat', 'Maize']
    farmers = {crop: create_farmers(num_farmers=len(params := [p for p in crop_params_list if p['name'] == crop]), 
                                  crop_params_list=params) for crop in crop_types}
    betas = {crop: calibrate_alpha_beta(farmers[crop], eta0, pw) for crop in crop_types}
    s1_0 = {crop: sum(farmer.s1 for farmer in farmers[crop]) for crop in crop_types}
    TW_0 = {crop: sum(farmer.TW for farmer in farmers[crop]) for crop in crop_types}
    return crop_types, farmers, betas, s1_0, TW_0

def create_eta_list(eta0, step=0.02):
    """创建效率提升序列"""
    eta_list = np.arange(eta0, 1, step)
    eta_list = np.append(eta_list, 1)
    return np.round(eta_list, 2)

def run_simulation(crop_types, farmers, eta_list, pw):
    """运行土地扩张模拟"""
    w_star_range = {crop: [] for crop in crop_types}  
    s1_total = {crop: [] for crop in crop_types}
    TW_total = {crop: [] for crop in crop_types}
    yield_total = {crop: [] for crop in crop_types}
    
    for type in crop_types:
        for eta in eta_list:
            print(f"正在模拟{type}作物，效率提升至{eta}")
            total_land, total_water, total_yield, w_star = land_expansion(farmers[type], eta, pw)
            w_star_range[type].append([w_star[0], w_star[1], w_star[2]])
            s1_total[type].append(total_land)
            TW_total[type].append(total_water)
            yield_total[type].append(total_yield)
    TW = [TW_total['Wheat'][i] + TW_total['Maize'][i] + TW_total['Rice'][i] for i in range(len(eta_list))]
    # 找到TW中最大值的索引对应的eta
    max_TW_index = np.argmax(TW)
    max_TW_eta = eta_list[max_TW_index]
    max_TW = TW[max_TW_index]
    return TW, max_TW_eta, max_TW

# 敏感性分析功能
def sensitivity_analysis(eta0, eta_list, pw, crop_params_file, sheet_name, perturbation_percent=0.05):
    """
    进行敏感性分析，分析各参数对total_sum的影响
    
    参数:
    eta0: 初始灌溉效率
    pw: 水价
    crop_params_file: 作物参数Excel文件名
    sheet_name: Excel表格名称
    perturbation_percent: 参数扰动百分比，默认5%
    
    返回:
    sensitivity_results: 敏感性分析结果字典
    """
    # 基准情况
    crop_types, farmers_base, betas_base, s1_0, TW_0 = initialize_simulation(eta0, pw, crop_params_file, sheet_name)
    TW_base, max_TW_eta_base, max_TW_base = run_simulation(crop_types, farmers_base, eta_list, pw=pw)
    
    print(f"基准情况下的效率阈值: {max_TW_eta_base}, 最大总用水量: {max_TW_base}")
    
    eta_sensitivity_results = {}
    TW_sensitivity_results = {}
    
    # 1. 分析水价pw的影响
    print("分析水价pw的敏感性...")
    pw_high = pw * (1 + perturbation_percent)
    pw_low = pw * (1 - perturbation_percent)
    
    # 高水价情况
    crop_types, farmers_high, betas_high, _, _ = initialize_simulation(eta0, pw_high, crop_params_file, sheet_name)
    TW_high, max_TW_eta_high, max_TW_high = run_simulation(crop_types, farmers_high, eta_list, pw=pw_high)
    
    # 低水价情况
    crop_types, farmers_low, betas_low, _, _ = initialize_simulation(eta0, pw_low, crop_params_file, sheet_name)
    TW_low, max_TW_eta_low, max_TW_low = run_simulation(crop_types, farmers_low, eta_list, pw=pw_low)
    
    eta_sensitivity_results['pw'] = {
        'high': max_TW_eta_high,
        'low': max_TW_eta_low,
        'change_high': (max_TW_eta_high - max_TW_eta_base) / max_TW_eta_base * 100,
        'change_low': (max_TW_eta_low - max_TW_eta_base) / max_TW_eta_base * 100
    }
    TW_sensitivity_results['pw'] = {
        'high': max_TW_high,
        'low': max_TW_low,
        'change_high': (max_TW_high - max_TW_base) / max_TW_base * 100,
        'change_low': (max_TW_low - max_TW_base) / max_TW_base * 100
    }
    
    # 2. 分析作物参数的影响
    crop_params_list = read_crop_params(crop_params_file, sheet_name)
    
    # 分别分析a, b, c, pc参数
    for param in ['a', 'b', 'c', 'pc']:
        print(f"分析参数{param}的敏感性...")
        
        # 创建高值参数列表
        crop_params_high = []
        for params in crop_params_list:
            params_high = params.copy()
            params_high[param] = params[param] * (1 + perturbation_percent)
            crop_params_high.append(params_high)
        
        # 创建低值参数列表
        crop_params_low = []
        for params in crop_params_list:
            params_low = params.copy()
            params_low[param] = params[param] * (1 - perturbation_percent)
            crop_params_low.append(params_low)
        
        # 高值情况
        farmers_high = {}
        for crop in crop_types:
            params_crop = [p for p in crop_params_high if p['name'] == crop]
            farmers_high[crop] = create_farmers(num_farmers=len(params_crop), crop_params_list=params_crop)
            calibrate_alpha_beta(farmers_high[crop], eta0, pw)
        
        TW_high, max_TW_eta_high, max_TW_high = run_simulation(crop_types, farmers_high, eta_list, pw=pw)
        
        # 低值情况
        farmers_low = {}
        for crop in crop_types:
            params_crop = [p for p in crop_params_low if p['name'] == crop]
            farmers_low[crop] = create_farmers(num_farmers=len(params_crop), crop_params_list=params_crop)
            calibrate_alpha_beta(farmers_low[crop], eta0, pw)
        
        TW_low, max_TW_eta_low, max_TW_low = run_simulation(crop_types, farmers_low, eta_list, pw=pw)
        
        eta_sensitivity_results[param] = {
            'high': max_TW_eta_high,
            'low': max_TW_eta_low,
            'change_high': (max_TW_eta_high - max_TW_eta_base) / max_TW_eta_base * 100,
            'change_low': (max_TW_eta_low - max_TW_eta_base) / max_TW_eta_base * 100
        }

        TW_sensitivity_results[param] = {
            'high': max_TW_high,
            'low': max_TW_low,
            'change_high': (max_TW_high - max_TW_base) / max_TW_base * 100,
            'change_low': (max_TW_low - max_TW_base) / max_TW_base * 100
        }

    # 3. 分析cost_min和cost_max的影响（同时变化）
    print("分析cost_min和cost_max的敏感性...")
    
    # 创建高值参数列表
    crop_params_high = []
    for params in crop_params_list:
        params_high = params.copy()
        params_high['cost_min'] = params['cost_min'] * (1 + perturbation_percent)
        params_high['cost_max'] = params['cost_max'] * (1 + perturbation_percent)
        params_high['cost'] = params['cost'] * (1 + perturbation_percent)
        crop_params_high.append(params_high)
    
    # 创建低值参数列表
    crop_params_low = []
    for params in crop_params_list:
        params_low = params.copy()
        params_low['cost_min'] = params['cost_min'] * (1 - perturbation_percent)
        params_low['cost_max'] = params['cost_max'] * (1 - perturbation_percent)
        params_low['cost'] = params['cost'] * (1 - perturbation_percent)
        crop_params_low.append(params_low)
    
    # 高值情况
    farmers_high = {}
    for crop in crop_types:
        params_crop = [p for p in crop_params_high if p['name'] == crop]
        farmers_high[crop] = create_farmers(num_farmers=len(params_crop), crop_params_list=params_crop)
        calibrate_alpha_beta(farmers_high[crop], eta0, pw)
    
    TW_high, max_TW_eta_high, max_TW_high = run_simulation(crop_types, farmers_high, eta_list, pw=pw)
    
    # 低值情况
    farmers_low = {}
    for crop in crop_types:
        params_crop = [p for p in crop_params_low if p['name'] == crop]
        farmers_low[crop] = create_farmers(num_farmers=len(params_crop), crop_params_list=params_crop)
        calibrate_alpha_beta(farmers_low[crop], eta0, pw)
    
    TW_low, max_TW_eta_low, max_TW_low = run_simulation(crop_types, farmers_low, eta_list, pw=pw)
    
    eta_sensitivity_results['alpha'] = {
        'high': max_TW_eta_high,
        'low': max_TW_eta_low,
        'change_high': (max_TW_eta_high - max_TW_eta_base) / max_TW_eta_base * 100,
        'change_low': (max_TW_eta_low - max_TW_eta_base) / max_TW_eta_base * 100
    }

    TW_sensitivity_results['alpha'] = {
        'high': max_TW_high,
        'low': max_TW_low,
        'change_high': (max_TW_high - max_TW_base) / max_TW_base * 100,
        'change_low': (max_TW_low - max_TW_base) / max_TW_base * 100
    }
    return eta_sensitivity_results, TW_sensitivity_results, max_TW_eta_base, max_TW_base

def plot_tornado_diagram(eta_sensitivity_results, TW_sensitivity_results, max_TW_eta_base, max_TW_base):
    """
    绘制龙卷风图展示敏感性分析结果
    """
    import matplotlib.pyplot as plt
    import os
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 创建左右分布的两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    
    # 准备效率阈值数据
    parameters = list(eta_sensitivity_results.keys())
    eta_changes_high = [eta_sensitivity_results[param]['change_high'] for param in parameters]
    eta_changes_low = [eta_sensitivity_results[param]['change_low'] for param in parameters]
    
    # 准备总用水量数据
    tw_changes_high = [TW_sensitivity_results[param]['change_high'] for param in parameters]
    tw_changes_low = [TW_sensitivity_results[param]['change_low'] for param in parameters]
    
    # 计算敏感性指数（取绝对值最大的变化）
    eta_sensitivity_indices = []
    tw_sensitivity_indices = []
    for i, param in enumerate(parameters):
        eta_max_change = max(abs(eta_changes_high[i]), abs(eta_changes_low[i]))
        tw_max_change = max(abs(tw_changes_high[i]), abs(tw_changes_low[i]))
        eta_sensitivity_indices.append(eta_max_change)
        tw_sensitivity_indices.append(tw_max_change)
    
    # 按敏感性指数排序
    eta_sorted_indices = sorted(range(len(eta_sensitivity_indices)), key=lambda i: eta_sensitivity_indices[i], reverse=True)
    tw_sorted_indices = sorted(range(len(tw_sensitivity_indices)), key=lambda i: tw_sensitivity_indices[i], reverse=True)
    
    eta_parameters_sorted = [parameters[i] for i in eta_sorted_indices]
    tw_parameters_sorted = [parameters[i] for i in tw_sorted_indices]
    
    # 转换参数标签为LaTeX格式
    def convert_to_latex(param):
        if param == 'alpha':
            return '$\\alpha$'
        elif param == 'beta':
            return '$\\beta$'
        elif param == 'pw':
            return '$p_w$'
        elif param == 'pc':
            return '$p_c$'
        elif param == 'eta':
            return '$\\eta$'
        elif param == 'a':
            return '$a$'
        elif param == 'b':
            return '$b$'
        elif param == 'c':
            return '$c$'
        else:
            return f'${param}$'
    
    eta_parameters_latex = [convert_to_latex(param) for param in eta_parameters_sorted]
    tw_parameters_latex = [convert_to_latex(param) for param in tw_parameters_sorted]
    
    eta_changes_high_sorted = [eta_changes_high[i] for i in eta_sorted_indices]
    eta_changes_low_sorted = [eta_changes_low[i] for i in eta_sorted_indices]
    tw_changes_high_sorted = [tw_changes_high[i] for i in tw_sorted_indices]
    tw_changes_low_sorted = [tw_changes_low[i] for i in tw_sorted_indices]
    
    # 绘制效率阈值的龙卷风图（左图）
    y_pos_eta = np.arange(len(eta_parameters_sorted))
    ax1.barh(y_pos_eta, eta_changes_high_sorted, color='royalblue', alpha=0.6, label='+5%')
    ax1.barh(y_pos_eta, eta_changes_low_sorted, color='tomato', alpha=0.6, label='-5%')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_yticks(y_pos_eta)
    ax1.set_yticklabels(eta_parameters_latex)
    ax1.set_xlabel('Relative variation (%)')
    ax1.set_title(f'(a) Sensitivity of efficiency threshold ($η^*$)')
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签（效率阈值）
    for i, (high, low) in enumerate(zip(eta_changes_high_sorted, eta_changes_low_sorted)):
        ax1.text(high/2, i, f'{high:.1f}%', va='center', ha='center', fontsize=14)
        ax1.text(low/2, i, f'{low:.1f}%', va='center', ha='center', fontsize=14)
    
    # 绘制总用水量的龙卷风图（右图）
    y_pos_tw = np.arange(len(tw_parameters_sorted))
    ax2.barh(y_pos_tw, tw_changes_high_sorted, color='royalblue', alpha=0.6, label='+5%')
    ax2.barh(y_pos_tw, tw_changes_low_sorted, color='tomato', alpha=0.6, label='-5%')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_yticks(y_pos_tw)
    ax2.set_yticklabels(tw_parameters_latex)
    ax2.set_xlabel('Relative variation (%)')
    ax2.set_title(f'(b) Sensitivity of maximum total water (TW)')
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签（总用水量）
    for i, (high, low) in enumerate(zip(tw_changes_high_sorted, tw_changes_low_sorted)):
        if i >=3:
            pos_high = 3
            pos_low = -3
        else:
            pos_high = high/2
            pos_low = low/2
        ax2.text(pos_high, i, f'{high:.1f}%', va='center', ha='center', fontsize=14)
        ax2.text(pos_low, i, f'{low:.1f}%', va='center', ha='center', fontsize=14)
    
    plt.tight_layout()

    # 创建保存目录
    save_dir = r'F:\AquaCrop-Paradox\Figures'  # 请根据实际路径修改
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'sensitivity_analysis_tornado.tif'), dpi=900, format='tif', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'sensitivity_analysis_tornado.pdf'), dpi=900, format='pdf', bbox_inches='tight')
    plt.show()
    
    return fig

#%%
# 主程序
eta0 = 0.5   
eta1 = 0.82
pw = 0.08
# 指定读取文件的名字
crop_params_file = "sub_crop_params.xlsx"
sheet_name = "sub"
eta_list = create_eta_list(eta0)
# 运行敏感性分析
print("开始敏感性分析...")
eta_sensitivity_results, TW_sensitivity_results, max_TW_eta_base, max_TW_base = sensitivity_analysis(eta0, eta_list, pw, crop_params_file, sheet_name)

# 打印结果
print("\n敏感性分析结果:")
print("=" * 50)
# 同步输出eta_sensitivity_results和TW_sensitivity_results
for param in eta_sensitivity_results.keys():
    print(f"{param}-效率阈值:")
    print(f"  +5%变化: {eta_sensitivity_results[param]['change_high']:.2f}%")
    print(f"  -5%变化: {eta_sensitivity_results[param]['change_low']:.2f}%")
    print(f"  高值结果: {eta_sensitivity_results[param]['high']:.2f}")
    print(f"  低值结果: {eta_sensitivity_results[param]['low']:.2f}")
    print()
    print(f"{param}-总用水量:")
    print(f"  +5%变化: {TW_sensitivity_results[param]['change_high']:.2f}%")
    print(f"  -5%变化: {TW_sensitivity_results[param]['change_low']:.2f}%")
    print(f"  高值结果: {TW_sensitivity_results[param]['high']:.2f}")
    print(f"  低值结果: {TW_sensitivity_results[param]['low']:.2f}")
    print()
#%% 绘制龙卷风图
print("绘制龙卷风图...")
fig = plot_tornado_diagram(eta_sensitivity_results, TW_sensitivity_results, max_TW_eta_base, max_TW_base)
