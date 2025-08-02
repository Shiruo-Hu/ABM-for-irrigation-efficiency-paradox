import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd

class Farmer:
    _instances = []  # 静态列表，用于跟踪所有Farmer实例
    
    def __init__(self, crop_params):
        self.w_star = 0    # 最优用水量(m^3/ha)
        self.Y_star = 0    # 最优产量(kg/ha)
        self.index = crop_params['index']
        self.s0 = crop_params['area']*100       # 初始自己的土地面积(ha)
        self.s1 = crop_params['area']*100       # 当前自己的土地面积(ha)
        self.s2 = 0        # 预期可接受的土地扩张面积(ha)
        self.status = 'not expand'      # 土地扩张状态
        self.marginal_profit = 0        # 开发土地的边际利润
        self.TW = self.w_star * self.s1
        # 产量函数的参数：yield = -a*w^2 + b*w + c
        # 转换系数abc使得其适应单位m^3/ha（默认拟合的作物水分产量函数是用mm为单位）
        self.type = crop_params['name']
        self.a = -crop_params['a']/100   
        self.b = crop_params['b']/10
        self.c = crop_params['c']  
        self.pc = crop_params['pc']
        self.cost = crop_params['cost']
        self.cost_min = crop_params['cost_min']
        self.cost_max = crop_params['cost_max']
        self.alpha = 0
        self.beta = 0
        self.benefit = 0
        self.totalyield = 0
        # 将当前实例添加到实例列表中
        Farmer._instances.append(self)

    def optimal_irrgate(self, eta, pw):
        """利润最大化时的灌溉"""
        self.w_star = (self.b*self.pc*eta - pw) / (2*self.a*self.pc*eta**2)
        w = self.w_star*eta
        self.Y_star = -self.a*w**2 + self.b*w + self.c
        self.marginal_profit = self.pc * self.Y_star - pw * self.w_star
        self.TW = self.w_star * self.s1
        #print(f"{self.index}号农民种植作物{self.type}的边际收益：{self.marginal_profit}")
    
    def observe_land(self, farmers):
        """观察和自己种植类型相同的农民的土地总和"""
        total_land = sum(farmer.s1 for farmer in farmers)
        marginal_cost = self.alpha * self.beta * total_land ** (self.beta-1)
        self.s2 = (self.marginal_profit / (self.alpha * self.beta))**(1/(self.beta-1))
        if self.s2 > total_land + 1 and self.marginal_profit > marginal_cost + 1:
            self.status = 'expand'
        else:
            self.status = 'not expand'
        #print(f"总土地面积{total_land}")
        #print(f"{self.index}号农民种植作物{self.type}的边际收益：{self.marginal_profit}，边际成本：{marginal_cost},{self.alpha},{self.beta}，状态：{self.status}")
        return total_land
    


def create_farmers(num_farmers, crop_params_list):
    """
    创建多个具有不同作物参数的农民
    """
    farmers_list = []
    for i in range(num_farmers):
        crop_params = crop_params_list[i]
        # 创建农民实例
        farmer = Farmer(crop_params=crop_params)
        farmers_list.append(farmer)
    
    return farmers_list

def read_crop_params(file_name, sheet_name):
    """
    读取作物参数Excel文件，并转换为字典列表格式
    
    参数:
    file_name: Excel文件名
    
    返回:
    crop_params_list: 包含作物参数的字典列表
    """
    # 读取Excel文件，确保将index列作为字符串读取以保留前导零
    df = pd.read_excel(file_name, sheet_name=sheet_name,  dtype={'index': str})
    
    # 转换为字典列表
    crop_params_list = []
    for _, row in df.iterrows():
        # 将每行数据转换为字典
        crop_param = {
            'index': row['index'],  
            'name': row['name'],
            'a': row['a'],
            'b': row['b'],
            'c': row['c'],
            'area': row['area'],
            'pc': row['pc'],
            'cost': row['cost'],
            'cost_min': row['cost_min'],
            'cost_max': row['cost_max']
        }
        crop_params_list.append(crop_param)
    
    return crop_params_list

def calibrate_alpha_beta(farmers, eta, pw):
    """
    通过对比成本是否落在范围内，校准beta值
    """
    # 计算总土地面积
    total_land = sum(farmer.s0 for farmer in farmers)
    
    # 计算每个农民的最优灌溉和边际利润，找出beta范围
    for farmer in farmers:
        farmer.optimal_irrgate(eta, pw)
        # 解方程 max_profit = alpha * beta * total_land^(beta-1)
        # 使用数值方法求解非线性方程
        def equation(beta_val):
            left_side = farmer.marginal_profit
            right_side = farmer.cost * beta_val * (total_land**(beta_val-1))
            return abs(left_side - right_side)  # 返回方程两边差值的绝对值    
        # 使用scipy的minimize_scalar求解beta使equation接近0
        result = minimize_scalar(equation, bounds=(1.001, 10.0), method='bounded')
        farmer.beta = result.x  
    # 找出w_star最大的农民的index
    max_w_star_index = np.argmax([farmer.w_star for farmer in farmers])
    max_w_star_farmer = farmers[max_w_star_index]
    print(f"w_star最大的农民: {max_w_star_farmer.index}, w_star: {max_w_star_farmer.w_star}")
    # 将所有beta值收集到一个列表中
    beta_values = [farmer.beta for farmer in farmers]
    beta_avg = sum(beta_values) / len(beta_values)
    beta_min = min(beta_values)
    beta_max = max(beta_values)
    beta_median = np.median(beta_values)
    print(f"灌区beta均值: {beta_avg}, 最小值: {beta_min}, 最大值: {beta_max}, 中位数: {beta_median}")

    # 在beta区间范围内找出最接近平均种植成本的beta
    # beta_min, beta_max保留三位小数
    beta_min = round(beta_min, 3)
    beta_max = round(beta_max, 3)
    beta_range = np.arange(beta_min, beta_max, 0.001)
    satisfied_num = np.zeros(len(beta_range), dtype=int)

    for i, beta in enumerate(beta_range):
        for farmer in farmers:
            farmer.alpha = farmer.marginal_profit / (beta * total_land**(beta-1))           
            if farmer.alpha <= 1.1*farmer.cost_max and farmer.alpha >= 0.9*farmer.cost_min:
                satisfied_num[i] += 1

    # 找出satisfied_num最大的beta的索引
    max_satisfied_index = int(np.argmax(satisfied_num))
    satisfied_rate = satisfied_num[max_satisfied_index] / len(farmers)
    calibrated_beta = beta_range[max_satisfied_index]
    # 固定beta值，校准alpha值
    for farmer in farmers:
        farmer.beta = calibrated_beta
        farmer.alpha = farmer.marginal_profit / (calibrated_beta * total_land**(calibrated_beta-1)) 
    print(f"率定的beta: {calibrated_beta}, 种植成本满足率: {satisfied_rate}")

    return calibrated_beta,  satisfied_rate

def adjust_beta(farmers, delta_beta, eta, pw):
    for farmer in farmers:
        farmer.beta = farmer.beta * (1+delta_beta)
        # 重新计算初始土地分布
        farmer.s1 = 0
        land_expansion(farmers, eta, pw)
    return farmers

        

def land_expansion(farmers, eta, pw):
    i = 0
    for farmer in farmers:
        farmer.optimal_irrgate(eta, pw)
    # 循环直到所有农民的土地扩张状态为不扩张
    while True:
        i += 1
        for farmer in farmers:
            farmer.TW = farmer.w_star * farmer.s1
            farmer.totalyield = farmer.Y_star * farmer.s1
            farmer.benefit = farmer.marginal_profit * farmer.s1 - farmer.alpha * farmer.s1**(farmer.beta)
            total_land = farmer.observe_land(farmers)            
        if all(farmer.status == 'not expand' for farmer in farmers):
            #print(f"扩张{i}次后，扩张完毕")
            total_land = sum(farmer.s1 for farmer in farmers)
            total_water = sum(farmer.TW for farmer in farmers)
            total_yield = sum(farmer.totalyield for farmer in farmers)
            # 返回w_star中的最大值中值最小值
            w_star_list = [farmer.w_star for farmer in farmers]
            w_star_max = max(w_star_list)
            w_star_min = min(w_star_list)
            w_star_median = np.median(w_star_list)
            w_star = [w_star_min, w_star_median, w_star_max]
            #print(f"扩张后总土地面积: {total_land}, 总用水量: {total_water}, 总产量: {total_yield}")
            break
        else:
            # 找出土地扩张状态为扩张的农民
            expand_farmers = [farmer for farmer in farmers if farmer.status == 'expand']
            # 找出其中预期扩张面积的最小值
            min_expand_land = min(farmer.s2 for farmer in expand_farmers)
            expand_land = min_expand_land - total_land
            # 根据每个农民的初始土地面积分配扩张的土地
            initial_total_land = sum(farmer.s0 for farmer in expand_farmers)
            for farmer in expand_farmers:
                farmer.s1 = farmer.s1 + expand_land * farmer.s0 / initial_total_land
    # 返回关键信息
    return total_land, total_water, total_yield, w_star



def export_farmers_to_excel(farmers, output_file='farmers_data.xlsx'):
    """
    将farmers中的数据导出到Excel文件
    
    参数:
    farmers: 包含Farmer对象的列表
    output_file: 输出Excel文件名 (默认为'farmers_data.xlsx')
    """
    # 创建用于保存数据的列表
    data = []
    for farmer in farmers:
        # 为每个farmer创建一行数据
        row = {
            'index': farmer.index,
            'type': farmer.type,
            'area': farmer.s1,
            'w_star': farmer.w_star,
            'Y_star': farmer.Y_star,
            'marginal_profit': farmer.marginal_profit,
            'alpha': farmer.alpha,
            'cost': farmer.cost,
            'cost_min': farmer.cost_min,
            'cost_max': farmer.cost_max,
            'beta': farmer.beta,
            'a': farmer.a,
            'b': farmer.b,
            'c': farmer.c,
            'pc': farmer.pc
        }
        data.append(row)
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(data)
    
    # 保存到Excel文件
    df.to_excel(output_file, index=False)
    print(f"数据已成功导出到 {output_file}，共 {len(data)} 行。")
    return df

def calibrate_alpha(farmers, eta, pw, beta):
    """
    校准alpha值
    """
    # 计算总土地面积
    total_land = sum(farmer.s0 for farmer in Farmer._instances)
    
    # 计算每个农民的最优灌溉和边际利润
    for farmer in farmers:
        farmer.beta = beta
        # 解方程 max_profit = alpha * beta * total_land^(beta-1)
        farmer.alpha = farmer.marginal_profit / (farmer.beta * total_land**(farmer.beta-1))
        # 验证结果
        marginal_cost = farmer.alpha * farmer.beta * (total_land**(farmer.beta-1))
        print(f"{farmer.index}号农民校准alpha: {farmer.alpha}，初始alpha: {farmer.cost}")
        print(f"误差: {abs(farmer.marginal_profit - marginal_cost)}")

def calibrate_beta_through_cost(farmers, eta, pw):
    """
    通过对比成本的绝对值，校准beta值
    """
    # 计算总土地面积
    total_land = sum(farmer.s0 for farmer in Farmer._instances)
    
    # 计算每个农民的最优灌溉和边际利润，找出beta范围
    for farmer in farmers:
        farmer.optimal_irrgate(eta, pw)
        # 解方程 max_profit = alpha * beta * total_land^(beta-1)
        # 使用数值方法求解非线性方程
        def equation(beta_val):
            left_side = farmer.marginal_profit
            right_side = farmer.cost * beta_val * (total_land**(beta_val-1))
            return abs(left_side - right_side)  # 返回方程两边差值的绝对值    
        # 使用scipy的minimize_scalar求解beta使equation接近0
        result = minimize_scalar(equation, bounds=(1.01, 10.0), method='bounded')
        farmer.beta = result.x  
    # 将所有beta值收集到一个列表中
    beta_values = [farmer.beta for farmer in farmers]
    beta_avg = sum(beta_values) / len(beta_values)
    beta_min = min(beta_values)
    beta_max = max(beta_values)
    beta_median = np.median(beta_values)
    print(f"灌区beta均值: {beta_avg}, 最小值: {beta_min}, 最大值: {beta_max}, 中位数: {beta_median}")

    # 在beta区间范围内找出最接近平均种植成本的beta
    # beta_min, beta_max保留三位小数
    beta_min = round(beta_min, 3)
    beta_max = round(beta_max, 3)
    rmse = []
    mae = []
    beta_range = np.arange(beta_min, beta_max, 0.001)
    
    for beta in beta_range:
        alpha_values = []
        cost_values = []
        for farmer in farmers:
            farmer.alpha = farmer.marginal_profit / (beta * total_land**(beta-1))
            alpha_values.append(farmer.alpha)
            cost_values.append(farmer.cost)
        # 求所有farmer.alpha与farmer.cost的MAE和均方根误差
        mae.append(np.mean(np.abs(np.array(alpha_values) - np.array(cost_values))))
        mse = np.mean((np.array(alpha_values) - np.array(cost_values))**2)
        rmse.append(np.sqrt(mse))

    # 找出rmse最小的beta的索引
    min_rmse_index = int(np.argmin(rmse))
    min_rmse = rmse[min_rmse_index]
    min_mae_index = int(np.argmin(mae))
    min_mae = mae[min_mae_index]
    calibrated_beta_rmse = beta_range[min_rmse_index]
    calibrated_beta_mae = beta_range[min_mae_index]

    # 求cost的均值，比较当前索引下(RMSE或MAE) 值是否 <15% 均值
    cost_avg = np.mean(cost_values) 

    if min_rmse  < 0.2  * cost_avg or min_mae < 0.2 * cost_avg:
        print(f"满足小于20%条件：RMSE: {min_rmse}, MAE: {min_mae}, 20%均值: {0.2*cost_avg}")
    else:
        print(f"不满足小于20%条件:RMSE: {min_rmse}, MAE: {min_mae}, 20%均值: {0.2*cost_avg}")
        
    print(f"种植成本rmse最小的beta: {calibrated_beta_rmse}, mae最小的beta: {calibrated_beta_mae}")

    return calibrated_beta_rmse,  min_rmse, calibrated_beta_mae, min_mae
