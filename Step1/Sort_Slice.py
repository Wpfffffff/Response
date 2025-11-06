import numpy as np
import time
import os
def LeadingPoint(points):
    """
    Subroutine to extract the leading edge point.
    Called by the main routine :func:`_Pt2Foil()`
    """
    #提取前缘点的y坐标
    min_index = np.argmin(points[:, 1])
    LeadingPoint = points[min_index, 2]
    return LeadingPoint
def middlePoint(points):
    """
    Subroutine to extract the trailing edge point and sort the trailing edge points based on y value.
    Called by the main routine :func:`_Pt2Foil()`
    """
    ##提取后缘点对应的行，并按照y值大小排序，得到后缘中点————用于区分上下翼面
    # 取出第二列数据等于 1 的行
    trailing_rows = points[points[:, 1] == 1]
    airfoil_lines = points[points[:, 1] != 1]
    # 按照第三列数据从大到小排序
    sorted_rows = trailing_rows[np.argsort(trailing_rows[:, 1])[::-1]]
    # k = sorted_rows.shape[0]
    # Point = (sorted_rows[0,2]+sorted_rows[k-1,2])/2
    Point = sorted_rows[0,2] #RAE2822初始翼型，可以直接以上翼面后缘点区分上下翼面
    return Point,sorted_rows,airfoil_lines

def split(data_lines,middlePoint,LeadingPoint):
    """
    Subroutine to split the airfoil into upper and lower surfaces based on the LeadingPoint and MiddlePoint
    Called by the main routine :func:`_Pt2Foil()`
    """
    #按照前缘点、后缘中点分离上下翼面
    #前半部分按照前缘点分成上下翼面
    leading_rows = data_lines[data_lines[:, 1] <= 0.5]
    upper_1 = leading_rows[leading_rows[:, 2] > LeadingPoint]  #上翼面
    lower_1 = leading_rows[leading_rows[:, 2] <= LeadingPoint]  # 下翼面
    #后半部分按照后缘中点分成上下翼面
    trailing_rows = data_lines[data_lines[:, 1] > 0.5]
    upper_2 = trailing_rows[trailing_rows[:, 2] > middlePoint]  #上翼面
    lower_2 = trailing_rows[trailing_rows[:, 2] <= middlePoint]  # 下翼面
    upper = np.vstack((upper_1,upper_2))
    lower = np.vstack((lower_1,lower_2))
    upper_sorted = upper[np.argsort(upper[:, 1])]
    lower_sorted = lower[np.argsort(lower[:, 1])[::-1]]
    airfoil_sorted = np.vstack((lower_sorted,upper_sorted))

    return upper_sorted,lower_sorted,airfoil_sorted

def JudgeSort(airfoil_slice):
    #通过Δx和Δy判断排序异常的点
    # 创建一个布尔数组以标记异常点
    is_anomaly = np.zeros(len(airfoil_slice), dtype=bool)
    anomal_id = []
    # 遍历点集，检查每对相邻点
    for i in range(1, len(airfoil_slice)):
        # 当前点和前一个点
        current_point = airfoil_slice[i]
        previous_point = airfoil_slice[i - 1]
        # 计算 Δx 和 Δy
        delta_x = current_point[1] - previous_point[1]
        delta_y = current_point[2] - previous_point[2]
        # 判断是否为异常点
        # if abs(delta_y)/abs(previous_point[2]) > 3 * (abs(delta_x)/abs(previous_point[1])):
        if abs(delta_y)*10 > 3 * abs(delta_x) and airfoil_slice[i, 1] > 0.5:
            is_anomaly[i] = True  # 标记为异常点
            anomal_id.append(i)
    # 提取异常点
    anomalies = airfoil_slice[is_anomaly]
    return  anomalies, np.array(anomal_id,dtype=int)

def anomalies_sort(upper_sorted,lower_sorted):
    anomalies_u, anomalu_id =  JudgeSort(upper_sorted)
    anomalies_d, anoamld_id =  JudgeSort(lower_sorted)
    upper_sorted =  delete_specific_points(upper_sorted, anomalu_id)
    lower_sorted =  delete_specific_points(lower_sorted, anoamld_id)
    # np.savetxt(self.outputDir+'/'+'upper_del_anomal.dat',upper_sorted)
    # np.savetxt(self.outputDir+'/'+'lower_del_anomal.dat',lower_sorted)
    if anomalies_u.size > 0: # anomaly upper surface
        anomalset = anomalies_u
    elif anomalies_d.size > 0: # anomaly lower surface
        anomalset = anomalies_d
    group1 = np.sum(anomalset[1::2,2]) # odd index
    # print('group1:',anomalset[1::2])
    group2 = np.sum(anomalset[0::2,2]) # even index
    # print('group2:',anomalset[0::2])
    # print('group1:',anomalset[1::2,2],'group2:',anomalset[0::2,2])
    # print('group1 sum:',group1,'group2 sum:',group2)
    if group1 > group2:
        # print('group1>group2')
        lower_sorted_add = np.vstack((lower_sorted,anomalset[0::2]))
        upper_sorted_add = np.vstack((upper_sorted,anomalset[1::2]))
    elif group1 < group2:
        # print('group1<group2')
        lower_sorted_add = np.vstack((lower_sorted,anomalset[1::2]))
        upper_sorted_add = np.vstack((upper_sorted,anomalset[0::2]))

    upper_sorted_add_1 = upper_sorted_add[np.argsort(upper_sorted_add[:, 1])]
    lower_sorted_add_1 = lower_sorted_add[np.argsort(lower_sorted_add[:, 1])[::-1]]
    airfoil_sorted = np.vstack((lower_sorted_add_1,upper_sorted_add_1))
    # if self.verbose:
        # np.savetxt(self.outputDir+'/'+'anomalies_u.dat',anomalies_u)
        # np.savetxt(self.outputDir+'/'+'anomalies_d.dat',anomalies_d)
        # np.savetxt(self.outputDir+'/'+'upper_sorted_add.dat',upper_sorted_add)
        # np.savetxt(self.outputDir+'/'+'lower_sorted_add.dat',lower_sorted_add)
        # print('!!! GeoDR warning !!! : upper anomaly points:',anomalies_u.shape)
        # print(anomalies_u)
        # print('!!! GeoDR warning !!! : lower anomaly points:',anomalies_d.shape)
        # print(anomalies_d)
    return airfoil_sorted
def remove_duplicates(arr):
    """ 
    Remove duplicate points from an array.
    """
    # pick No.2 and No.3 columns
    duplicated_cols = arr[:, 1:3]
    
    # find the unique indices of the duplicated points
    count = 0
    duplicated_pos = []
    for i in range(1, len(duplicated_cols)):
        if np.array_equal(duplicated_cols[i], duplicated_cols[i - 1]):
            duplicated_pos.append(i)
            count += 1 
    
    # pick the first column values of the unique indices
    duppoints = arr[np.array(duplicated_pos)]
    
        # remove the duplicated points
    unique_arr =  delete_specific_points(arr, duplicated_pos)
    
    return unique_arr, duppoints

def delete_specific_points(arr, indices):
    mask = np.ones(len(arr), dtype=bool)
    mask[indices] = False
    # remove the duplicated points
    return_arr = arr[mask]
    return return_arr

def Pt2Foil(allpoints):
    """
    Sort the airfoil coordinates from all the surface points gathered from all processors.
    Note: this function is only called on master processor.
    """
    #将行号添加新的一列
    num_rows = allpoints.shape[0]
    row_numbers = np.arange(1, num_rows + 1).reshape(-1, 1)
    allpoints_withNumber = np.hstack((row_numbers, allpoints))

    #————————————————————————————找到前缘点——————————————————————————————————————————————————————————
    #———————————————————————————————————————————————————————————————————————————————————————————————
    LeadingPoint_1 =  LeadingPoint(allpoints_withNumber)
    #————————————————————————————找到后缘中点，按照y值排序后的后缘点————————————————————————————————————
    #————————————————————————————————————————————————————————————————————————————————————————————————
    MiddlePoint_1,TE_Point_1,airfoil_lines_1 =  middlePoint(allpoints_withNumber)
    #————————————————————————————按照前缘点、后缘中点分成上下两部分—————————————————————————————————————
    #————————————————————————————————————————————————————————————————————————————————————————————————
    upper_sorted_1,lower_sorted_1 =  split(airfoil_lines_1,MiddlePoint_1,LeadingPoint_1)
    #————————————————————————————针对上下翼面判断异常排序的点—————————————————————————————————————
    #———————————————————————————————————————————————————————————————————————————————————————————
    airfoil_slice1_judge =  anomalies_sort(upper_sorted_1,lower_sorted_1)

    # if self.verbose :
        # print('# GeoDR: airfoil_slice1_includedup shape:', airfoil_slice1_judge.shape)
        # print('# GeoDR: airfoil_slice2_includedup shape:', airfoil_slice2_judge.shape)
        # np.savetxt(self.outputDir+'/'+'airfoil_slice1_includedup.dat',airfoil_slice1_judge)
        # np.savetxt(self.outputDir+'/'+'airfoil_slice2_includedup.dat',airfoil_slice2_judge)
    #————————————————————————————去掉重复点——————————————————————————————————————————
    airfoil_slice1_judge, duppoints1 = remove_duplicates(airfoil_slice1_judge)

    # add trailing edge points (top and bottom) to the airfoil slices (now we don't need trailing edge points)
    # airfoil_slice1_judge = np.vstack((TE_Point_1[-1,:], airfoil_slice1_judge, TE_Point_1[0,:]))

    # if self.verbose :
        
    #     print('# GeoDR: LeadingPoint_1:',LeadingPoint_1,'LeadingPoint_2:',LeadingPoint_2)
    #     print('# GeoDR: MiddlePoint_TE_1:',MiddlePoint_1,'MiddlePoint_TE_2:',MiddlePoint_2)
    #     print('# GeoDR: airfoil_slice1 shape:',airfoil_slice1_judge.shape)
    #     print('# GeoDR: airfoil_slice2 shape:',airfoil_slice2_judge.shape)
    #     print('# GeoDR: TE_Point_1 shape:',TE_Point_1.shape)
    #     print('# GeoDR: TE_Point_2 shape:',TE_Point_2.shape)
    #     np.savetxt(self.outputDir+'/'+'airfoil_slice1.dat',airfoil_slice1_judge)
    #     np.savetxt(self.outputDir+'/'+'airfoil_slice2.dat',airfoil_slice2_judge)
    #     np.savetxt(self.outputDir+'/'+'upper_sorted_1.dat',upper_sorted_1)
    #     np.savetxt(self.outputDir+'/'+'lower_sorted_1.dat',lower_sorted_1)
    #     np.savetxt(self.outputDir+'/'+'upper_sorted_2.dat',upper_sorted_2)
    #     np.savetxt(self.outputDir+'/'+'lower_sorted_2.dat',lower_sorted_2)
        
    return airfoil_slice1_judge,TE_Point_1,duppoints1

# 记录开始时间
start_time = time.time()

# # # #————————————————————对RAE2822初始翼型进行排序，得到各行的排列顺序————————————————————————————
fi = open('./output_zero/slice_1.dat','r')
lines = fi.readlines() # have read all the lines
for j,line in enumerate(lines):
    if 'Nodes ' in line:
        flagline = j
        nNodes = int(lines[j].split()[2])
# print('-> nNodes:',nNodes)
nVars=14
SliceSurfData= np.zeros((nNodes,nVars))
num = 0
for k in range(flagline+2,flagline+nNodes+2):
    if 'E-' in lines[k] or 'E+' in line[k]:
        SliceSurfData[num,:] = np.array(list(map(float, lines[k].split())))
        num += 1
fi.close()

AirfoilData_ini = SliceSurfData[:,0:3]
#将行号添加新的一列
num_rows = AirfoilData_ini.shape[0]
row_numbers = np.arange(1, num_rows + 1).reshape(-1, 1)
allpoints_withNumber = np.hstack((row_numbers, AirfoilData_ini))

LeadingPoint_1 =  LeadingPoint(allpoints_withNumber)

MiddlePoint_1,TE_Point_1,airfoil_lines_1 =  middlePoint(allpoints_withNumber)

upper_sorted_1,lower_sorted_1,airfoil_sorted =  split(airfoil_lines_1,MiddlePoint_1,LeadingPoint_1)

# airfoil_sorted_remove, duppoints1 = remove_duplicates(airfoil_sorted)

# serialNumber = airfoil_sorted_remove[:,0]
serialNumber = airfoil_sorted[:,0]
serialNumber_int = serialNumber.astype(int)  #行的顺序

np.savetxt('./output_sorted/rae2822_upper.dat',upper_sorted_1)
np.savetxt('./output_sorted/rae2822_lower.dat',lower_sorted_1)
np.savetxt('./output_sorted/rae2822_sorted.dat',airfoil_sorted)
# np.savetxt('./output_sorted/rae2822_sorted_remove.dat',airfoil_sorted_remove)
np.savetxt('./output_sorted/serialNumber_int.dat',serialNumber_int)

# # #————————————————————对采样点排序————————————————————————————————————————————————————————
N = 10000
for i in range(N):
    file_path = f'./output/slice_{i+1}.dat'  # 用f-string更简洁
    
    
    if not os.path.exists(file_path):
        # print(f"文件不存在：{file_path}，跳过")
        continue  # 跳过当前循环，进入下一次
    
    # 文件存在时才执行打开操作
    with open(file_path, 'r') as fi:  # 建议用with语句，自动关闭文件
 

        lines = fi.readlines() # have read all the lines
        for j,line in enumerate(lines):
            if 'Nodes ' in line:
                flagline = j
                nNodes = int(lines[j].split()[2])
        # print('-> nNodes:',nNodes)
        nVars=14
        SliceSurfData= np.zeros((nNodes,nVars))
        num = 0
        for k in range(flagline+2,flagline+nNodes+2):
            if 'E-' in lines[k] or 'E+' in line[k]:
                SliceSurfData[num,:] = np.array(list(map(float, lines[k].split())))
                num += 1
        fi.close()
    
        AirfoilData = SliceSurfData[:,0:2]
        AirfoilData_sorted = AirfoilData[serialNumber_int-1]  #将采样点按照已有行号进行排列
        np.savetxt('./output_sorted/Airfoil_'+str(i)+'.dat',AirfoilData_sorted)


# ————————————————————————存储为.npz格式,(N_sample,n_airfoil,2)————————————————————————————————



N_sample = 10000 #样本点个数
n_airfoil = 197 #翼型离散点个数
data_array = np.empty((N_sample, n_airfoil, 2))
airfoil_ini = np.loadtxt('./output_sorted/rae2822_sorted.dat')
airfoil_ini_1 = airfoil_ini[:,1:3]
data_array[0] = airfoil_ini_1


for k in range(N_sample - 1):
    file_path = f'./output_sorted/Airfoil_{k}.dat'
    if os.path.exists(file_path):
        data_array[k+1] = np.loadtxt(file_path)
np.savez('RAE2822_data.npz', data=data_array)

# #——————————————————————————检查翼型采样结果的维度————————————————————————————————————
Airfoils = np.load('./RAE2822_data.npz')
print("Arrays in the .npz file:", Airfoils.files)
data = Airfoils['data']
print(data.shape)

# 记录结束时间
end_time = time.time()

# 计算并打印运行时间（保留2位小数）
run_time = end_time - start_time
print(f"程序运行时间：{run_time:.2f} 秒")