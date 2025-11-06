from pyDOE import lhs
import numpy as np
import time
# # define dimension

# 记录开始时间
start_time = time.time()



dim = 16  #维度 
Num = 10000 #采样点并不是全满足要求
lhd = lhs(dim, samples=Num)

lower_bounds = np.array([-0.02,  -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, 0, 0,  -0.02, -0.02,  -0.02, -0.02, -0.02, -0.02, -0.02]) #(FFD)
upper_bounds = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0,  0,  0.02, 0.02,  0.02, 0.02, 0.02, 0.02, 0.02])
samples = lower_bounds + lhd * (upper_bounds - lower_bounds)
np.savetxt('./samples_dim16_N_time'+str(Num)+'.dat',samples)


# 记录结束时间
end_time = time.time()

# 计算并打印运行时间（保留2位小数）
run_time = end_time - start_time
print(f"程序运行时间：{run_time:.2f} 秒")
