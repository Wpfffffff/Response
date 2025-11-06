# ======================================================================
#         Import modules
# ======================================================================
# rst Imports (beg)
import os
import numpy as np
import argparse
import pygpc
import time
import math
from mpi4py import MPI
from multipoint import multiPointSparse
from adflow2 import ADFLOW
from idwarp import *
from pygeo import *
from pyspline import *
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyDOE import lhs
from baseclasses import *
import warnings
import time

# 记录开始时间
start_time = time.time()
folder_problem = "output_problem"

# 创建文件夹（仅创建单级目录，已存在会抛异常）
if not os.path.exists(folder_problem):
    os.mkdir(folder_problem)



warnings.filterwarnings("ignore")

# rst Imports (end)
parser = argparse.ArgumentParser()
parser.add_argument("--ngrid",type=int,default=600)
args = parser.parse_args()
n_grid = args.ngrid
# FFDFile = 'ffd_r1.xyz'
FFDFile = 'ffd.xyz'
gridFile = '../../rae2822.cgns'
# gridFile = 'rae2822_n250.cgns'
# gridFile = 'rae2822.cgns'
# fname = 'Geo_Random_samples'+str(n_grid)+'.dat'
fname = '/fs2/home/yangtihao/someone/wupengfei/LiuXing_test/LunWen/Step0/samples_zero.dat'



# ======================================================================
#         Create samples set
# ======================================================================

# define dimension
dim = 16
samples = np.loadtxt(fname)
# Num = 50
# lhd = lhs(dim, samples=Num)
# # lower_bounds=-0.02
# # upper_bounds= 0.02
# lower_bounds = np.array([-0.006, -0.006, -0.008, -0.008, -0.02, -0.01, -0.01, -0.006,-0.01, -0.02, -0.02, -0.02, -0.01, -0.01, -0.01, -0.008])
# upper_bounds = np.array([0.006, 0.006, 0.008, 0.008, 0.02, 0.01, 0.01, 0.006, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.008])#delete the first lines is feasible
# # lower_bounds = np.array([-0.008, -0.008, -0.02, -0.02, -0.02, -0.02, -0.02, -0.01,-0.008, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02])
# # upper_bounds = np.array([0.008, 0.008, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01,-0.008, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
# samples = lower_bounds + lhd * (upper_bounds - lower_bounds)
# np.savetxt('./samples.dat',samples)

nGroup = 1
nProcPerGroup = MPI.COMM_WORLD.size
MP = multiPointSparse(MPI.COMM_WORLD)
MP.addProcessorSet("cruise", nMembers=nGroup, memberSizes=MPI.COMM_WORLD.size)
comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()
# if comm.rank == 0:
#     if not os.path.exists('ffd'):
#         os.mkdir('ffd')
#     if not os.path.exists('grid'):
#         os.mkdir('grid')

output_directory = './output_zero'
if MPI.COMM_WORLD.rank == 0:
    if(not(os.path.exists(output_directory))):
        os.mkdir(output_directory)

aeroOptions = {
    # Common Parameters
    "gridFile": gridFile,
    "outputDirectory": output_directory,
    # Physics Parameters
    "equationType": "RANS",
    "smoother": "DADI",
    "MGCycle": "sg",
    "nCycles": 20000,
    "monitorvariables": ["resrho", "cl", "cd", "cmz", "yplus"],
    'surfaceVariables':['vx','vy','vz','rho','P','mach','cp','temp'],
    'volumevariables':['resrho','rhoe','vort','mach','dist','eddy','resturb','cp','ptloss'],
    "useNKSolver": True, # turn off by chenyifu
    "useanksolver": True,
    "nsubiterturb": 15, # increase from 10 to 15 by chenyifu
    "liftIndex": 2,
    "infchangecorrection": True,
    # Convergence Parameters
    "L2Convergence": 1e-10,
    "L2ConvergenceCoarse": 1e-2,
    # Adjoint Parameters
    "adjointSolver": "GMRES",
    "adjointL2Convergence": 1e-12,
    "ADPC": True,
    "adjointMaxIter": 5000,
    "adjointSubspaceSize": 400,
    "ILUFill": 3,
    "ASMOverlap": 3,
    "outerPreconIts": 3,
    "NKSubSpaceSize": 400,
    "NKASMOverlap": 4,
    "NKPCILUFill": 4,
    "NKJacobianLag": 5,
    "nkswitchtol": 1e-6,
    "nkouterpreconits": 3,
    "NKInnerPreConIts": 3,
    "writeSurfaceSolution": True,
    "writeVolumeSolution": True,
    'writeTecplotSurfaceSolution':True,
    "frozenTurbulence": False,
    "restartADjoint": False,

    # add by chenyifu
    # "rkreset": True,
    # "nrkreset": 100,
    # "useqcr":True,
}

#Laplacian smoothing algorithm 
def laplacian_smoothing(y_coordinates, n_neighbor, alpha=0.5):
    num_points = len(y_coordinates)
    smoothed_y_coordinates = y_coordinates.copy()

    for i in range(1, num_points):  # 从第2个控制点到倒数第2个控制点
        if i != num_points/2:
            if i == 9:
                smoothed_y_coordinates[i] = alpha*y_coordinates[i]+((1-alpha)/n_neighbor)*(-1*y_coordinates[0]+y_coordinates[i+1])
            if i == 15:
                smoothed_y_coordinates[i] = alpha*y_coordinates[i]+((1-alpha)/n_neighbor)*(y_coordinates[i-1]+(-1*y_coordinates[int(num_points/2)]))
            else:
                smoothed_y_coordinates[i] = alpha*y_coordinates[i]+((1-alpha)/n_neighbor)*(y_coordinates[i-1]+y_coordinates[i+1])
    
    return smoothed_y_coordinates

# # Create solver
CFDSolver = ADFLOW(options=aeroOptions, comm=comm)


spanDirection = 'z'
span = 1
pos  = np.array([0.5])*span
CFDSolver.addSlices(spanDirection,pos,sliceType='absolute')

meshOptions = {
    'gridFile': gridFile,
    'fileType': 'CGNS'
    # 'warpType':'algebraic',
    }
mesh = USMesh(options=meshOptions, comm=comm)
CFDSolver.setMesh(mesh)

ap = AeroProblem(name='RAE2822',
    mach=0.8,
    reynolds=7.93e6,
    reynoldsLength = 1.0,
    T = 286.72 ,
    alpha=0.135227,
    areaRef=1.0,
    chordRef=1.0,
    xRef=0.25,yRef=0.0,zRef=0.0,
    evalFuncs=['cl','cd','cmz']
)

Thick_ini_1 = np.array([0.00801409, 0.0809715,  0.10726156, 0.11979794, 0.11854259, 0.10369998,
 0.07949045, 0.05120131, 0.02398148, 0.00513562, 0.00801409, 0.0809715,
 0.10726156, 0.11979794, 0.11854259, 0.10369998, 0.07949045, 0.05120131,
 0.02398148, 0.00513562])*0.9


# Thick_ini_2 = np.array([0.00801409, 0.0559453, 0.07698409, 0.00801409, 0.0559453, 0.07698409])*0.9

Volum_lower = 0.07785*0.95
Volum_upper = 0.07785*1.02

FeasibleThickCon_1 = []
# FeasibleThickCon_2 = []
FeasibleVolumCon = []
DV_feasible = []
num = 1
volum_V0 = []
# This is the initial surface coordinates DO NOT USE THIS NAME !!
coords0_ini = mesh.getSurfaceCoordinates()
# This is the initial design variable values DO NOT USE THIS NAME !!

for i in range(samples.shape[0]):
    exec(open('./setup_geometry.py').read())
    CFDSolver.setDVGeo(DVGeo)
    xDV = DVGeo.getValues()
    # coords0 = mesh.getSurfaceCoordinates()
    ptSetName = 'allSurfs'
    CFDSolver.DVGeo.addPointSet(coords0_ini,ptSetName)
    # mesh.setSurfaceCoordinates(DVGeo.update(ptSetName))
    CFDSolver.setAeroProblem(ap)
    exec(open('./setup_constraints.py').read())
    # xDV = DVGeo.getValues() # get the initial value of design variables which is also the delta value of FFD control point
    # print('xDVini :',xDV)
    xDVd = samples[i,:]
    xDVd_smooth = laplacian_smoothing(xDVd,2,alpha=0.5)
    xDVd_smooth_2 = laplacian_smoothing(xDVd_smooth,2,alpha=0.5)

    for j in range(dim):
        xDV['shape'][j] = xDVd_smooth_2[j]

    CFDSolver.DVGeo.setDesignVars(xDV)

    Confuncs = {}
    DVCon.evalFunctions(Confuncs)
    # DVCon.writeTecplot('./output/constraints_'+str(i)+'.dat')
    if comm.rank == 0:
        print('Confuncs:thickness and volum:',Confuncs)

    # mesh.setSurfaceCoordinates(DVGeo.update(ptSetName))
    # mesh.warpMesh()
    # CFDSolver.updateGeometryInfo()
    CFDSolver.setAeroProblem(ap)
    
    Thickness_1 = Confuncs['DVCon1_thickness_constraints_0']
    # Thickness_2 = Confuncs['DVCon1_thickness_constraints_1']
    volum = Confuncs['DVCon1_volume_constraint_0']
    if comm.rank == 0:
        print(DVCon.constraints)
        print('V0:',DVCon.constraints['volCon']['DVCon1_volume_constraint_0'].V0)
        print('D0:',DVCon.constraints['thickCon']['DVCon1_thickness_constraints_0'].D0)
        # print('D0:',DVCon.constraints['thickCon']['DVCon1_thickness_constraints_1'].D0)
        # print(DVCon.constraints['volCon']['DVCon1_volume_constraint_0'].coords)
        print('Thickness_1_'+str(i+1)+':',Thickness_1)
        # print('Thickness_2_'+str(i+1)+':',Thickness_2)
        print('volum_'+str(i+1)+':',volum)

    
    # if np.all(Thickness_1 >= Thick_ini_1):
    #     print("Thick YES")
    # if np.all(volum >= Volum_lower):
    #     print("Volum Lower YES")
    # if np.all(volum <= Volum_upper):
    #     print("Volum Upper YES")
    if np.all(Thickness_1 >= Thick_ini_1) and np.all((volum >= Volum_lower) & (volum <= Volum_upper)):#0.3/1/1.01
    # mesh.writeGrid('./output/grid'+str(i)+'.cgns')
    # DVGeo.writeTecplot('./output/ffd'+str(i)+'.dat') 
        CFDSolver.writeSlicesFile('./output_zero/slice_'+str(num)+'.dat')
        num += 1
        FeasibleThickCon_1.append(Thickness_1)
        # FeasibleThickCon_2.append(Thickness_2)
        FeasibleVolumCon.append(volum)
        DV_feasible.append(xDVd)
        volum_V0.append(DVCon.constraints['volCon']['DVCon1_volume_constraint_0'].V0)
    else:
        CFDSolver.writeSlicesFile('./output_problem/slice_'+str(num)+'.dat')
        num += 1

np.savetxt('./output_zero/Thickness_1.dat',FeasibleThickCon_1)
# np.savetxt('./output/Thickness_2.dat',FeasibleThickCon_2)
np.savetxt('./output_zero/Volum.dat',FeasibleVolumCon)#输出满足约束采样点的体积和厚度结果
np.savetxt('./output_zero/DV_feasible.dat',DV_feasible)
np.savetxt('./output_zero/volum_V0.dat',volum_V0)

# 记录结束时间
end_time = time.time()

# 计算并打印运行时间（保留2位小数）
run_time = end_time - start_time
print(f"程序运行时间：{run_time:.2f} 秒")
# def set_y(val, geo):
#     C = geo.extractCoef('axis')   

#     # C[0,1] += -1.0*val[0]
#     for i in range(0,int(nCP/2)-1):
#       C[i,1] += val[i]
#       C[i + nCP, 1] += val[i]
    
#     C[int(nCP/2)-1, 1] += -1*val[int(nCP/2) - 1]
#     C[int(nCP/2), 1] += 1*val[int(nCP/2)- 1]
    
#     for i in range(int(nCP/2) + 1, nCP):
#       C[i,1] += val[i-1]
#       C[i + nCP, 1] += val[i-1]
#     # C[nCP-1,1] +=  val[0]
    
#     for i in range(1,nCP+1):
#     	C[i-1+nCP,1] = C[i-1,1]
    
#     #print('C:	',C)
#     geo.restoreCoef(C, 'axis')


