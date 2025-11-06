# ======================================================================
#         DVConstraint Setup
# ====================================================================== 
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)

# Only ADflow has the getTriangulatedSurface Function
DVCon.setSurface(CFDSolver.getTriangulatedMeshSurface())
le=0.001
leList = [[le    , 0, 0.0+le], [le    , 0, 1.0-le]]
teList = [[1.0-le, 0, 0.0+le], [1.0-le, 0, 1.0-le]]
# Thickness constraints
# DVCon.addVolumeConstraint(leList, teList, 2, 25, lower=1.0,upper=30.0)
DVCon.addThicknessConstraints2D(leList, teList, 2, 10, lower=0.3, scaled=False)
#DVCon.addThicknessConstraints2Dvector(leList, teList, 2, 50, lower=0.1)

#Leading Edge
teList_2 = [[0.1, 0, 0.0+le], [0.1, 0, 1.0-le]]
DVCon.addThicknessConstraints2D(leList, teList_2, 2, 3, lower=0.3, scaled=False)

ledis=0.001
sydis=0.001
leList = [[ledis    , 0, sydis], [ledis    , 0, 1.0-sydis]]
teList = [[1.0-ledis, 0, sydis], [1.0-ledis, 0, 1.0-sydis]]     # Thickness constraints
#DVCon.addThicknessConstraints2D(leList, teList, 2, 50, lower=1.0)     #volume constraints
DVCon.addVolumeConstraint(leList, teList, 2, 25, lower=1.0,upper=30.0,scaled=False)

# lIndex = DVGeo.getLocalIndex(0) #LETE function assumes that in spanwise there should be more than 2 FFD points.so this 2D case got error.
# indSetA = []
# indSetB = []
# for k in range(0,1):
#     indSetA.append(lIndex[0, 0, k])
#     indSetB.append(lIndex[0, 1, k])
# for k in range(0,1):
#     indSetA.append(lIndex[-1, 0, k])
#     indSetB.append(lIndex[-1, 1, k])

# #make variables in two symmetry plane to be the same
# #print lIndex


# lIndex = DVGeo.getLocalIndex(0)
# indSetA = []
# indSetB = []
# for i in range(lIndex.shape[0]):
#     indSetA.append(lIndex[i, 0, 0])
#     indSetB.append(lIndex[i, 0, 1])
# for i in range(lIndex.shape[0]):
#     indSetA.append(lIndex[i, 1, 0])
#     indSetB.append(lIndex[i, 1, 1])
# DVCon.addLinearConstraintsShape(indSetA, indSetB,factorA=1.0, factorB=-1.0,lower=0, upper=0)


# if comm.rank == 0:
#     fileName = './results/constraints.dat'
#     DVCon.writeTecplot(fileName)

# Confuncs = {}
# DVCon.evalFunctions(Confuncs)
# DVCon.writeTecplot('./output/constraints_'+str(i)+'.dat')
# if comm.rank == 0:
#     print('Confuncs:thickness and volum:',Confuncs)
