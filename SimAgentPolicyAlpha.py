#%% 讨论成本政策
import importlib
import Agent
importlib.reload(Agent)
from Agent import create_farmers, calibrate_alpha_beta, read_crop_params, land_expansion
import numpy as np
import plot_utils
importlib.reload(plot_utils)
from plot_utils import  plot_pw_water_use_heatmap

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
    
    # 为每个作物类型创建新的farmers副本
    farmers_copy = {crop: farmers[crop].copy() for crop in crop_types}
    
    for type in crop_types:
        for eta in eta_list:
            print(f"正在模拟{type}作物，效率提升至{eta}")
            # 使用farmers的副本进行计算
            total_land, total_water, total_yield, w_star = land_expansion(farmers_copy[type], eta, pw)
            w_star_range[type].append([w_star[0], w_star[1], w_star[2]])
            s1_total[type].append(total_land)
            TW_total[type].append(total_water)
            yield_total[type].append(total_yield)
    
    return w_star_range, s1_total, TW_total, yield_total

# 主程序
eta0 = 0.5   
pw0 = 0.08
# 指定读取文件的名字
crop_params_file = "sub_crop_params.xlsx"
sheet_name = "sub"
# 创建情景序列
alpha_list = np.arange(0, 51, 5) 
eta_list = create_eta_list(eta0)

all_TW_total = {}
for alpha in alpha_list:
    print(f"\n正在模拟成本 alpha = {alpha:.0f}")
    crop_types, farmers, betas, s1_0, TW_0 = initialize_simulation(eta0, pw0, crop_params_file, sheet_name)
    # 遍历每种作物类型的农民
    for crop_type in crop_types:
        for farmer in farmers[crop_type]:
            alpha0 = farmer.alpha
            farmer.alpha = alpha + alpha0
    w_star_range, s1_total, TW_total, yield_total = run_simulation(crop_types, farmers, eta_list, pw0)
    all_TW_total[alpha] = TW_total


#%% 绘制图表
import plot_utils
importlib.reload(plot_utils)
from plot_utils import plot_alpha_water_use_heatmap

# 绘制热力图
print("\n绘制总用水量热力图")

total_water_matrix = plot_alpha_water_use_heatmap(eta_list, alpha_list, all_TW_total, crop_types)
# %%
