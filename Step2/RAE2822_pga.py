import os
import numpy as np
import time

# 记录开始时间
start_time = time.time()
# G2Aero
from g2aero.PGA import Grassmann_PGAspace, SPD_TangentSpace 
from g2aero.manifolds import ProductManifold, Dataset
from g2aero import SPD as spd
from g2aero import Grassmann as gr

#Plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import torch

######___训练数据集，得到PGA空间__##################
##################################################

#Read airfoil data from subdirectory
# this is a nice set of randomly generated CST shapes for wind turbine design provided by Andrew Glaws
# data = np.load('./RAE2822_data_NoConstraints_average_14points_smooth_v3.npz')
# print("文件中的键：", data.files)  # 输出类似 ['Vh', 'other_key']
shapes = np.load( './average/RAE2822_data.npz')['data']
print("Dataset:")
print(f"Shape of data = {shapes.shape}")
print(f"N_shapes = {shapes.shape[0]}")
print(f"n_landmarks in every shape = {shapes.shape[1]}")

# # # #landmark-affine (LA) standardization using SVD
shapes_gr, M, b = gr.landmark_affine_transform(shapes)



# # # # compute Karcher mean and run PGA to define coordinates
gr_pga, T = Grassmann_PGAspace.create_from_dataset(shapes, method='LA-transform')
# # # # print('Vh_shape:',gr_pga.Vh.shape)

# # # # # save the computed Grassmannian PGA space
gr_pga.save_to_file('./average/RAE2822_GPA.npz')

# ————————————————————————重构RAE2822,求解初始设计变量Var————————————————————————
gr_pga = Grassmann_PGAspace.load_from_file('./average/RAE2822_GPA.npz') #load的是得出的PGA
# # assign r as the dimension of the PGA shape
r = 18 # should always be less than or equal to 2*(n_landmarks - 2)
# # # # pick a shape based on index from the dataset
# j = 0 # should be less than or equal to N_shapes-1
# R = gr_pga.t[j,:r]
t = gr_pga.t[:, :r]
# 保存为文本文件（适用于小规模数据）
np.savetxt('pga_scores.txt', t, delimiter='\t', fmt='%.6f')
print("数据已保存至 pga_scores.txt")


# 记录结束时间
end_time = time.time()

# 计算并打印运行时间（保留2位小数）
run_time = end_time - start_time
print(f"程序运行时间：{run_time:.2f} 秒")


#设计空间的范围
max_values = np.max(t, axis=0)
min_values = np.min(t, axis=0)

space = np.vstack((max_values,min_values))

np.savetxt('./average/space.dat',space)

# # # # 计算奇异值的积分线（累积和）
# cumulative = np.cumsum(gr_pga.S[:18])
# # 归一化积分值到 0-1 范围
# cumulative_normalized = (cumulative - np.min(cumulative)) / (np.max(cumulative) - np.min(cumulative))

# fig, ax1 = plt.subplots(1, 1)
# x = np.arange(1, 19)
# # plt.stem(gr_pga.S[:16]**2)
# plt.stem(x,gr_pga.S[:18])
# # plt.yscale('log')
# plt.grid(True, which='both')
# plt.xlabel('eigenvalue index')
# plt.ylabel('eigenvalue')
# ax1.set_xlim(0,19)
# ax1.xaxis.set_ticks(np.arange(0, 19, 2))

# ax2 = ax1.twinx() 
# ax2.set_ylabel('Cumulative', color='k')
# ax2.plot(x,cumulative_normalized, linestyle='-', color='k')
# plt.axhline(y=0.95, linestyle='--',color='#c0392b')
# plt.axhline(y=0.9, linestyle=':',color='#c0392b')
# plt.axhline(y=0.999, linestyle='-.',color='#c0392b')

# plt.savefig('eigenvalue.png', dpi=300, bbox_inches='tight')