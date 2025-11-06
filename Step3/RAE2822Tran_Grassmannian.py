"""
RAE2822
-----------
Optimization for transonic foil <RAE2822> by Grassmannian manifold
author : chenyifu
Version: python 3 version
"""
# ======================================================================
#         Import modules
# ======================================================================
import os, sys, copy, time, shutil
import argparse
import numpy as np
from mpi4py import MPI
from baseclasses import *
from adflowtransi_geoDR import ADFLOW
# from idwarp import USMesh
from pywarp import *
from pygeoDR import *
from pyspline import *
from multipoint import *
from pyoptsparse import Optimization, OPT
from team import TransitionCalc
from pyaerotransi_geoDR import AeroTransi
# from pyGeoDimensionReduction import GeoDimensionReduction,geoDVManifold
from collections import OrderedDict

# Use Python's built-in Argument parser to get commandline options
parser = argparse.ArgumentParser()
parser.add_argument("--shape", help='Use shape variables', type=int,
                    default=0)
parser.add_argument("--opt", help="optimizer to use", type=str, default='snopt')
parser.add_argument('--optOptions', type=str, help="Options for the optimizer.", default="{}")#
parser.add_argument("--procs", help="number of processors", type=int, default=28)
args = parser.parse_args()

# ======================================================================
#         Specify parameters for caculation
# ======================================================================
# rst parameters (beg)
name = 'RAE2822'
grid_file = '../rae2822.cgns'
FFDFile = '../ffd_Noaverage.xyz'
GrassmanFile = '../Step2/Noaverage/RAE2822_PGA_data.npz'
output_aero =  './output_aero'
output_transi = './output_transi'
output_GeoDR = './output_GeoDR'

global iCFD
iCFD=0 

# output_directory =  './output'
outputDirectory = './results'
# saveRepositoryInfo(output_directory)

nFlowCases = 1
# mach number
mach = [0.73] # RAE2822
# angle of attack
alpha = [2.0]
# design CL coefficient
CL_star = [0.7827]
# reynold number
reynolds = [7.93e6]
# design Cm coefficient
CM_star = [0.045]
# Weight 
Weight = [1.0]

nDVGrassman = 10

reynoldsLength = 1.0
T = 286.72
# reference area
areaRef=1.
# reference chord
chordRef=1.

spanDirection = 'z'
# rst parameters (end)

# ======================================================================
#         Create multipoint communication object
# ======================================================================
# Creat aero/transition comms
nGroup = 1
nProcPerGroup = args.procs

npTransi = 1
npAero = MPI.COMM_WORLD.size - npTransi

MP = multiPointSparse(MPI.COMM_WORLD)
MP.addProcessorSet('cruise', nMembers=nGroup, memberSizes=nProcPerGroup)
gcomm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()

comm, flags = createGroups([npTransi, npAero], comm=gcomm)
aeroID = 1
transiID = 0

if(gcomm.rank == 0):
    if(not(os.path.exists('output_aero'))):
        os.mkdir('output_aero')
if(gcomm.rank == 0):
    if(not(os.path.exists('output_transi'))):
        os.mkdir('output_transi')
if(gcomm.rank == 0):
    if(not(os.path.exists('output_GeoDR'))):
        os.mkdir('output_GeoDR')
if(gcomm.rank == 0):
    if(not(os.path.exists('results'))):
        os.mkdir('results')


# ======================================================================
#         Options Set-up
# ======================================================================
aeroOptions = {
    # I/O Parameters
    'gridFile':grid_file,
    'outputDirectory':output_aero,
    'monitorvariables':['resrho','cl','cpu','cd','cmz','resturb','cdp','cdv'],
    'volumevariables':['resrho','Intermittency', 'cp', 'mach', 'temp', 'rhoe'],
    'surfacevariables':['cp','vx', 'vy','vz', 'mach','cfx', 'mach', 'rho', 'p', 'temp', 'cf', 'yplus','blank'],
    'writeTecplotSurfaceSolution':True,
	#'restartFile' : './fc_000_vol.cgns',

    # Physics Parameters
    'equationType':'RANS',
    'useqcr':True,

    # Solver Parameters
    'CFL':1.5,
    'CFLCoarse':1.25,
    'MGCycle':'SG',
    
    'rkreset': True,
    'nrkreset' : 100,

    # ANK Solver Parameters
    'useANKSolver':True,
    'nsubiterturb':15,
    'anksecondordswitchtol':1e-4, # increased for 30 deg
    # 'ankcflfactor': 4.0,
    # 'ankcoupledswitchtol':1e-5,

    # ANK yayun's setting
	'useANKSolver':True,
    # 'ankuseturbdadi':False,
    # 'ankturbkspdebug':True,

	'ankstepfactor' : 0.5,

	#'ankcoupledswitchtol' : 1e-6,
	'ankmaxiter' : 60,

    # NK Solver Parameters
    'useNKSolver':False,
    'nkswitchtol':1e-6,
    'nkadpc':True,
    # 'nkasmoverlap': 3, # for highly parallel
    'nkinnerpreconits': 2,
    'nkjacobianlag': 3,
    'nkouterpreconits': 3,
    'nkpcilufill': 2,
    'nksubspacesize': 100,

    # Termination Criteria
    'L2Convergence':1e-10,
    'L2ConvergenceCoarse':1e-2,
    'nCycles':5000,
    'useblockettes':False,
    
    
    # if use transition
    'ntransition':True, 
    'transi2dim':True,
    'useintermittency':True,
	'BoundaryLayerThickness':0.06,
    'usexyzlstate':True,
    # 'rkreset':True,
    'nrkreset':100,

    # Adjoint Parameters
    'setMonitor':False,
    'applyadjointpcsubspacesize':15,
    'adjointL2Convergence':1e-8,
    'ADPC':True,
    'adjointMaxIter': 1000,
    'adjointSubspaceSize':150, #150
    'ILUFill':2,
    'ASMOverlap':1,
    'outerPreconIts':3,

}
#rst ADflow aero options(end)

#rst transi options(beg)
transiOptions={
    #Relaxation Parameter 
    'outputDirectory':output_transi,
    'RelaxTr':  0.8,
    'RelaxLen': 1.0,
    'TrLimit':1.0,
    'isCompressible':False,#Incompressible or Compressible LST icomp = 1,
    'minerror': 1e-6,
    'isLaminarTransition':False,
    'entype':1,
    'usexyzlstate':True,
    # 'laminarseptype':'seppoint',
    'laminarseptype':'cpcrit',
    # 'partInfo':{'allWalls':[False,False,0.0,0.0]},
}
#rst transi options(end)

#rst aerotransi options(begin)
lfOptions ={
    'adjointRelTol':1e-7,
    'outputDir':output_aero,
    'adjointDamp0':0.9,
    'reltol':1.0e-7,
    'adjointsolver':'GS',
    'nlfiter':25,
}
#rst aerotransi options(end)

#rst mesh options(begin)
meshOptions = {
    'gridFile': grid_file,
    # 'warpType':'algebraic',
    }
#rst mesh options(end)

# ======================================================================
#         Set up Problems
# ======================================================================
#rst aeroProblems (begin)
aeroProblems = []
for i in range(nFlowCases):
    ap = AeroProblem(name=name+str(CL_star[i]),
        mach=mach[i],
        reynolds=reynolds[i],
        reynoldsLength = reynoldsLength,
        T = T ,
        alpha=alpha[i],
        areaRef=areaRef,
        chordRef=chordRef,
        xRef=0.25,yRef=0.0,zRef=0.0,
        evalFuncs=['cl','cd','cmz']
    )
    ap.addDV('alpha', value=alpha[i], lower=-4.0, upper=4.0, scale=0.1)
    aeroProblems.append(ap)


transiProblems = []
for i in range(nFlowCases):
    tp = TransiProblem(name=name+str(CL_star[i]),
        mach=mach[i],
        reynolds=reynolds[i]/reynoldsLength,
        T=T,
        TurbulentIntencity = 0.0007,
        nCritTS=9.0,
        nCritCF=8.1,
        spanDirection=spanDirection,
        )
    transiProblems.append(tp)

# atp = AeroTransiProblem(ap,tp)

aerotransiProblems = []
for i in range(nFlowCases):
    aerotransiProblems.append(AeroTransiProblem(aeroProblems[i],transiProblems[i]))
nAeroTransiCase_Cruise = len(aerotransiProblems)


# ======================================================================
#         Geometric Design Variable Set-up
# ======================================================================
# rst dvgeo (beg)
# Setup Geometry
# execfile('./common_files/setup_geometry.py') # python 2
exec(open('../Step1/setup_geometry.py').read())
# rst dvgeo (end)

# np.random.default_rng(seed=42)
# Var = np.random.normal(0, 1, size=(nDVGrassman))
# if(gcomm.rank == 0):
#     print('Var_random:',Var)
#     print('Var_random_shape:',Var.shape)
# np.savetxt('./Var_random.dat',Var)

upper = np.array([0.1465,   0.1423,  0.1060,   0.09760,  0.08122,  0.06928,  0.06323,  0.04358, 0.03547, 0.02155])
lower = np.array([-0.1551, -0.1416,  -0.1098,  -0.09873, -0.07076, -0.07790, -0.06688, -0.04159,  -0.03398, -0.02144])

Var = np.zeros(nDVGrassman)

iniGrassmanVal = Var #np.loadtxt('./var.dat') #np.array([0.0040,  0.0096,  0.0014, -0.0064, -0.0089,  0.0042,  0.0031,  0.0004]) #
GeoDR = GeoDimensionReduction(GrassmanFile,outputDir=output_GeoDR,comm=comm,verbose=False)
GeoDR.addManifoldDV('grassman',iniVal=iniGrassmanVal,lower=lower,upper=upper,scale=1.0)

# ======================================================================
#         Set up Solvers
# ======================================================================
if flags[aeroID]:
    CFDSolver = ADFLOW(options=aeroOptions,comm=comm)
    # CFDSolver.setDVGeo(DVGeo)
    CFDSolver.setGeoDR(GeoDR)

    span = 1
    pos  = np.array([0.5])*span
    CFDSolver.addSlices(spanDirection,pos,sliceType='absolute')
    transiSolver = None

if flags[transiID]:
    transiSolver = TransitionCalc(options=transiOptions,comm=comm)
    # TransiSolver DVGeo
    CFDSolver = None

AT = AeroTransi(CFDSolver, transiSolver,gcomm,options=lfOptions)

# ======================================================================
#         Mesh Warping Set-up
# ======================================================================
# rst warp (beg)
if flags[aeroID]:
    
    # mesh = USMesh(options=meshOptions, comm=comm)
    mesh = MBMesh(options=meshOptions, comm=comm)
    CFDSolver.setMesh(mesh)

# rst warp (end)

# ======================================================================
#         DVConstraint Setup
# ======================================================================
# NOTE:!!!!!!!!! VERY IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# constraints should be set after the mesh warping otherwise the mesh 
# won't be updated for constraints evaluation
# if flags[aeroID]:
#     # execfile('./common_files/setup_constraints.py') # python 2
#     exec(open('./common_files/setup_constraints.py').read())



# ======================================================================
#         Solve Functions:
# ======================================================================
def cruiseFuncs(x):

    GeoDR.iOpt += 1
    # x['grassman'] = np.random.rand(nDVGrassman)*0.001
    if MPI.COMM_WORLD.rank == 0:
        curOutputdir = GeoDR.outputDir + '/iOpt' + str(GeoDR.iOpt) + '/'
        if(not(os.path.exists(curOutputdir))):
            os.mkdir(curOutputdir)
    if MPI.COMM_WORLD.rank == 0:
        print('Design Variables:',x)
    funcs = {}
    # DVGeo.setDesignVars(x)
    # x = np.random.uniform(0, 1, size=nDVGrassman)
    # x = np.random.rand(nDVGrassman)
    GeoDR.setDesignVars(x)
    # if flags[aeroID]:
    #     DVCon.evalFunctions(funcs)
        # print('DVCon', DVCon)
    # if flags[transiID]:
        # transiSolver.setDesignVars(x)
    global iCFD
    for i in range(nAeroTransiCase_Cruise):
        if i%nGroup == ptID:
            aerotransiProblems[i].setDesignVars(x)
            # if flags[transiID]:
            #     transiSolver.setOption('initforcedtransiloc',[[[0.58505234883292367, 8.9015095428028865E-002,0.49999999999999994,5.6951992312207531E-002],[0.28608187913374883,-5.6632289210587911E-002,0.49999999999999994,3.3691451707747538E-002]]])
            AT(aerotransiProblems[i])           
            AT.evalFunctions(aerotransiProblems[i], funcs)
            # DVGeo.writeTecplot('./results/ffd.dat')
            if MPI.COMM_WORLD.rank == 0:
                # os.rename('./results/ffd.dat','./results/ffd'+str(iCFD)+str(i)+'.dat')
                os.chdir('./results')
                if(not(os.path.exists(str(iCFD)+str(i)))):
                    os.mkdir(str(iCFD)+str(i))
                shutil.move('../output_transi/allSideOut.dat','./'+str(iCFD)+str(i)+'/allSideOut.dat')
                shutil.move('../output_transi/transiLoc.dat','./'+str(iCFD)+str(i)+'/transiLoc.dat')
                shutil.move('../output_transi/bledge_info.dat','./'+str(iCFD)+str(i)+'/bledge_info.dat')
                shutil.move('../output_transi/nfactor_ts.dat','./'+str(iCFD)+str(i)+'/nfactor_ts.dat')
                os.chdir('../')
    iCFD=iCFD+1

    if MPI.COMM_WORLD.rank == 1:
        print('######',GeoDR.iOpt,'######')
        print('x and funcs:', x, funcs)

    return funcs


def cruiseFuncsSens(x, funcs):
    funcsSens = {}
    # funcsSensDVGeo = {}
    # if flags[aeroID]:
    #     DVCon.evalFunctionsSens(funcsSens)

    for i in range(nAeroTransiCase_Cruise):
        if i%nGroup == ptID:
            AT.evalFunctionsSens(aerotransiProblems[i], funcsSens,['cl','cd','cmz'])
    
        # dCondPt = DVCon.evalFunctionsSens(funcsSensDVGeo)
        # dCondPt = {'Volume1':[],'Thickness1':[],}
        # if MPI.COMM_WORLD.rank == 1:# aero root
        #     print('dCondPt',dCondPt)
        #     dIdx = GeoDR.totalSensitivity(dCondPt,...) # [1x8]
        # MPI.COMM_WORLD.Bcast(dIdx, root=1)
        # funcsSens['DVCon1_thickness_constraints_0'] = dIdx
    if MPI.COMM_WORLD.rank ==0:
        print('######',GeoDR.iOpt,'######')
        print('funcsSens', funcsSens)
    return funcsSens



def objCon(funcs):
    # Assemble the objective and any additional constraints:
    funcs['obj'] = 0.0
    for i in range(nAeroTransiCase_Cruise):
        atp = aerotransiProblems[i]
        funcs['obj'] += funcs[atp['cd']]*Weight[i]  
        funcs['cl_con_'+atp.name] = funcs[atp['cl']] - CL_star[i]
        # if(i == 0):
        #     funcs['cmz_con_'+atp.name] = funcs[atp['cmz']] - CM_star[i]
    if MPI.COMM_WORLD.rank == 0:
        print('funcs',funcs)
    return funcs

# ======================================================================
#               Options for optimization
# ======================================================================
# Options settins for optimization
optOptions = {
    'Major iterations limit':10000,
    'Minor iterations limit':1000000,
    'Iterations limit':1000000,
    'Major step limit':0.01, 
    'Major feasibility tolerance':1.0e-5, # target nonlinear constraint violation
    'Major optimality tolerance':1.0e-5, # target complementarity gap 
    'Minor feasibility tolerance':1.0e-5,
    'Verify level':0 ,                    # check on gradients : -1 means disable the check
    # 'Major step limit':0.2,                                                               
    # 'Nonderivative linesearch':None,
    'Function precision':5.0e-6,
    'Print file':outputDirectory + '/SNOPT_print.out',
    'Summary file':outputDirectory + '/SNOPT_summary.out',
    'Problem Type':'Minimize',
    #'Problem Type':'Maximize',
    }


# outputDirectory = args.output
# saveRepositoryInfo(outputDirectory)

# ======================================================================
#         Set-up Optimization Problem
# ======================================================================
optProb = Optimization('opt', MP.obj, comm=MPI.COMM_WORLD)

# Add variables from each aeroProblem
# for ap in aeroProblems:
#     ap.addVariablesPyOpt(optProb)

# Add DVGeo variables
# DVGeo.addVariablesPyOpt(optProb)
# Add GeoDR variables
GeoDR.addVariablesPyOpt(optProb)

# Add DVConstraint constraints
# if flags[aeroID]:
#     DVCon.addConstraintsPyOpt(optProb)
# ===========================================================================
# Setup Optimization Problem and define the objection and constrains function 
# ===========================================================================
for i in range(nAeroTransiCase_Cruise):
    atp = aerotransiProblems[i]
    atp.addVariablesPyOpt(optProb)
    if(i==0):
        optProb.addCon('cl_con_' +atp.name, lower=0.0, upper=0.08, scale=1.0)
    if(i>0):
        optProb.addCon('cl_con_' +atp.name, lower=0.0, upper=0.02, scale=1.0) 
    # if(i == 0):
        # optProb.addCon('cmz_con_' +atp.name, lower=-1.0, upper=0.0, scale=1.0)
# Add Objective
optProb.addObj('obj', scale=10000)

# The MP object needs the 'obj' and 'sens' function for each proc set,
# the optimization problem and what the objcon function is:
nCall = 0
MP.addProcSetObjFunc('cruise', cruiseFuncs)
MP.addProcSetSensFunc('cruise', cruiseFuncsSens)
MP.setObjCon(objCon)
MP.setOptProb(optProb)

# Make Instance of Optimizer
opt = OPT(args.opt, options=optOptions)

# optProb.setDVsFromHistory('snopt_hist.hst')
if MPI.COMM_WORLD.rank == 0:
    print('optProb', optProb)
optProb.printSparsity()
# Run Optimization
histFile = os.path.join(outputDirectory, '%s_hist.hst'%args.opt)
sol = opt(optProb, MP.sens, storeHistory=histFile)

