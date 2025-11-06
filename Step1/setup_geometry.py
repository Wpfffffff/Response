   
import numpy as np
# from pywarp import *
from pygeo import *
from pyspline import *
import warnings
warnings.filterwarnings("ignore")

nCP = 18
nPos = 16

DVGeo = DVGeometry(FFDFile)


coef = DVGeo.FFD.vols[0].coef.copy()
coef_top, coef_bottom = map(np.array, zip(*coef))
coef_new = np.concatenate((coef_top,np.flipud(coef_bottom)), axis=0)
coef = coef_new
#print('coef:  ',coef)
#print('coef12:  ',coef_top[:,0,1])
#print('coef13:  ',coef_top[:,1,1])



#print('coef:	',coef.shape)
nSpan = coef.shape[0]
ref = np.zeros((nSpan*2,3))

for k in range(nSpan):
    ref[k,0] = coef[k,0,0]
    ref[k,1] = coef[k,0,1]
    ref[k,2] = 0.0

    ref[k + nSpan,0] = coef[k,1,0]
    ref[k + nSpan,1] = coef[k,1,1]
    ref[k + nSpan,2] = 1.0

X = ref
#print('ref: ',ref)

c0 = Curve(X=X, k=2)
DVGeo.addRefAxis('axis', c0)

def set_y(val, geo):
  C = geo.extractCoef('axis')   
  # if MPI.COMM_WORLD.rank == 0:
  # print('C.shape:',C.shape)
  # print(C)
  num = 0

  # leading edge
  C[0,1] += -1.0*val[num]
  C[nCP-1,1] +=  val[num]
  C[nCP,1] = C[0,1]
  C[int(2*nCP)-1,1] = C[nCP-1,1]
  num = num + 1

  # nCP/2=8
  # internal lower surface [1,8)
  for i in range(1,int(nCP/2)-1):
    C[i,1] += val[num]
    C[i + nCP, 1] += val[num]
    num = num + 1
  
  # tail edge
  C[int(nCP/2)-1, 1] += -1*val[num]
  C[int(nCP/2), 1] += 1*val[num]
  C[int(nCP/2)-1+nCP,1] = C[int(nCP/2)-1,1]
  C[int(nCP/2)+nCP,1] = C[int(nCP/2),1]
  num = num + 1

  # internal upper surface [16,9)
  for i in range(nCP - 2, int(nCP/2),-1):
    C[i,1] += val[num]
    C[i + nCP, 1] += val[num]
    num = num + 1


  print('The number of DVs is :',num)
  geo.restoreCoef(C, 'axis')


DVGeo.addGlobalDV('shape', np.zeros(nPos), set_y, lower=-0.02, upper=0.02, scale=1e0)









