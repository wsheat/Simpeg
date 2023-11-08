import os
from pyexpat import model
from pydoc import describe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import mu_0, inch, foot
import os
#import ipywidgets
import time
from string import ascii_lowercase
from matplotlib import rcParams
from matplotlib import gridspec

import discretize
from discretize import utils
from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG import maps, utils
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver
import scipy.sparse as sp
from SimPEG import (
data,
optimization,
data_misfit,
regularization,
inverse_problem,
inversion,
directives,
Report,
)

import matplotlib as mpl
#plt.style.use("ggplot")

freq=2.0
#################################################################################################################################################
#Mesh
csx, ncx, npadx = 10, 16, 2
csz, ncz, npadz = 10, 16, 2
hx = utils.meshTensor([(csx,npadx,-1.3),(csx, ncx), (csx, npadx, 1.3)])
hz = utils.meshTensor([(csz, npadz, -1.3), (csz, ncz), (csz, npadz, 1.3)])
# define the Tensor mesh
mesh = discretize.TensorMesh([hx, hx,hz])
mesh.x0 = np.r_[
            0.0, -mesh.h[1].sum() / 2.0, -mesh.h[2].sum()
         ]
print(
f"The mesh has {mesh.nC} cells \n"
f" * x-extent: {mesh.nodes_x[-1]:1.1e} m\n"
f" * z-extent: [{mesh.nodes_z[0]:1.1e} m, {mesh.nodes_z[-1]:1.1e} m]")
mesh.plotGrid()
plt.show()
#################################################################################################################################################
# True model
actind=mesh.gridCC[:, 2] <0.0
'''
#Inclined sheet model
blk1 = utils.model_builder.getIndicesBlock(
np.r_[120,  -20, -60],
np.r_[140,  20, -70], 
mesh.gridCC
     )
blk2 = utils.model_builder.getIndicesBlock(
np.r_[110, -20, -70],
np.r_[130, 20, -80], 
mesh.gridCC
     )
blk3= utils.model_builder.getIndicesBlock(
np.r_[100,  -20, -80],
np.r_[120,  20, -90], 
mesh.gridCC
     )
blk4 = utils.model_builder.getIndicesBlock(
np.r_[90, -20, -90],
np.r_[110, 20, -100], 
mesh.gridCC
     )
blk5= utils.model_builder.getIndicesBlock(
np.r_[80,  -20, -100],
np.r_[100,  20, -110], 
mesh.gridCC
     )
blk6 = utils.model_builder.getIndicesBlock(
np.r_[70, -20, -110],
np.r_[90, 20, -120], 
mesh.gridCC
     )
blk7 = utils.model_builder.getIndicesBlock(
np.r_[60, -20, -120],
np.r_[80, 20, -130], 
mesh.gridCC
     )
'''
#Cuboid Model
blk1 = utils.model_builder.getIndicesBlock(
np.r_[40,  -20, -40],
np.r_[160,  20, -80], 
mesh.gridCC
     )
blk2 = utils.model_builder.getIndicesBlock(
np.r_[40, -20, -120],
np.r_[160, 20, -160], 
mesh.gridCC
     )
blk3 = utils.model_builder.getIndicesBlock(
np.r_[40, -20, -200],
np.r_[160, 20, -240], 
mesh.gridCC
     )

sigma_air = 1e-8
layer_inds = mesh.gridCC[:, 2] > -5.0
sigma = np.ones(mesh.nC) * 1.0 / 1e8
sigma[actind] = 1/ 10
sigma[blk1] = 1/100.0
sigma[blk2] = 1/100.0
sigma[blk3] = 1/1.0
rho = 1.0 / sigma
mtrue = np.log(rho[actind])

#################################################################################################################################################
#Survey
orientation = 'z'
src_z=np.linspace(-200, 0, 21)
src_loc2=np.zeros((len(src_z),3))
src_loc2[:,1]=-100
src_loc2[:,2]=src_z

src_z1=np.linspace(-200, 0, 21)
src_loc1=np.zeros((len(src_z1),3))
src_loc1[:,1]=100
src_loc1[:,2]=src_z1
src_loc1=np.concatenate((src_loc1,src_loc2))

rx_z=np.linspace(-200, 0,21)
rx_loc1=np.zeros((len(rx_z),3))
rx_loc1[:,0]=200
rx_loc1[:,1]=100
rx_loc1[:,2]=rx_z

rx_loc2=np.zeros((len(rx_z),3))
rx_loc2[:,0]=200
rx_loc2[:,1]=-100
rx_loc2[:,2]=rx_z
rx_loc=np.concatenate((rx_loc1,rx_loc2))

# fname = "F:/lxc/simpeg-main/examples/yueshu/2/src_loc.txt"
# np.savetxt(fname,src_loc2,fmt="%.2f", header='log(rho[actind])')
# fname = "F:/lxc/simpeg-main/examples/yueshu/2/rx_loc.txt"
# np.savetxt(fname,rx_loc1,fmt="%.2f", header='log(rho[actind])')

srcList1 = []
for isrc_z in src_loc1:
    src_loc3=isrc_z
    rx_real=fdem.Rx.PointElectricField(rx_loc, orientation='z', component='real')  #Point_e(PointElectricField)
    s_ind = utils.closestPoints(mesh, src_loc3, "Fx") #+ mesh.nEx
    de = np.zeros(mesh.nF, dtype=complex)
    de[s_ind] = 1.0/csz
    rxList=[rx_real]
    src1 = fdem.Src.RawVec_e(rxList, freq,de/mesh.face_areas)
    srcList1.append(src1)

survey2=fdem.Survey(srcList1)

mx=mesh.gridCC[:, 0]
my=mesh.gridCC[:, 1]
mz=mesh.gridCC[:, 2]
# fname = "F:/lxc/simpeg-main/examples/yueshu/2/sigma.txt"
# np.savetxt(fname,np.c_[mx,my,mz,rho],fmt="%.2f", header='log(rho[actind])')

actmap = maps.InjectActiveCells(mesh, indActive=actind, valInactive=np.log(1e8))
mapping = maps.ExpMap(mesh) * actmap
prob2=fdem.Simulation3DMagneticField(mesh,survey=survey2,rhoMap=mapping)#,Solver=Solver)
rel_err = 0.05
# Make synthetic data with 5% Gaussian noise
data02 = prob2.make_synthetic_data(mtrue, relative_error=rel_err,noise_floor=1e-11,add_noise=True)
sol32=data02.dclean

xref = 110.
zref = -50.
yref = -10.

xlim = [0., 300.]
ylim = [-150., 150.]
zlim = [-300., 0.]
indx = int(np.argmin(abs(mesh.cell_centers_x)))
indy = int(np.argmin(abs(mesh.cell_centers_y)))
indz = int(np.argmin(abs(mesh.cell_centers_z)))

fig, ax = plt.subplots(1,3, figsize = (12,10))
clim = [10, 100]
dat1 = mesh.plotSlice(rho, grid=False, ax=ax[0], ind=indy, normal='Y', cmap='RdBu_r')
dat2 = mesh.plotSlice(rho, grid=False, ax=ax[1], ind=indx, normal='X',cmap='RdBu_r')
dat3 = mesh.plotSlice(rho, grid=False, ax=ax[2], ind=indz,cmap='RdBu_r')
ax[0].set_xlim(xlim)
ax[0].set_ylim(zlim)
ax[1].set_xlim(ylim)
ax[1].set_ylim(zlim)
ax[2].set_xlim(xlim)
ax[2].set_ylim(ylim)
plt.tight_layout()
ax[0].set_aspect(1)
ax[1].set_aspect(1)
ax[2].set_aspect(1)
ax[0].set_title("Y={:1.0f}m".format(mesh.cell_centers_y[indy]))
ax[1].set_title("X={:=1.0f}m".format(mesh.cell_centers_x[indx]))
ax[2].set_title("Z={:1.0f}m".format(mesh.cell_centers_z[indz]))
#cb.set_label("$\sigma$ (S/m)")
plt.show()

##################################################################################################################################################
#Inversion
t = time.time()

relative=0.05
eps=10**(-3.2)
uncert = abs(data02.dobs) * relative + eps
dmis = data_misfit.L2DataMisfit(simulation=prob2, data=data02)
dmis.standard_deviation = uncert

reg = regularization.Sparse(mesh, mapping=maps.IdentityMap(mesh))
#reg.mref = np.zeros(nParam)
p = 2.0
qx = 2.0
qy=2.0
qz=2.0
reg.norms = [p,qx,qy,qz]

opt = optimization.InexactGaussNewton(maxIterCG=20, maxIter = 20)
opt.lower = 0.0
opt.remember("xc")
opt.tolG = 1e-15
opt.eps = 1e-15
# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
saveDict = directives.SaveOutputEveryIteration(save_txt=False)
beta_schedule= directives.BetaSchedule(coolingFactor=5, coolingRate=2)
IRLS = directives.Update_IRLS(max_irls_iterations=40, minGNiter=1, f_min_change=1e-4)
directives_list = [
                starting_beta,#directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
                beta_schedule,
                saveDict,
                IRLS,
            ]
inv = inversion.BaseInversion(inv_prob, directiveList=directives_list)
# Starting model
m0 = np.log(np.ones(mtrue.size) * 10)

mrec = inv.run(m0)
model1 = opt.recall("xc")
model1.append(mrec)
pred = []
for m in model1:
    pred.append(prob2.dpred(m))

print("\n Inversion Complete. Elapsed Time = {:1.2f} s".format(time.time() - t))
#############################################################################################

saveDict.plot_misfit_curves()
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
beta = np.array(saveDict.beta)
        # account for the 1/2 so phi_d* = nD
phi_d = 2*np.array(saveDict.phi_d)
phi_m = 2*np.array(saveDict.phi_m)
nD = saveDict.survey[0].nD
i_target = None
i_target = 0
while i_target < len(phi_d) and phi_d[i_target] > nD:
        i_target += 1

ax[0].semilogx(beta, phi_d)
ax[0].semilogx(beta, np.ones_like(beta)*nD, '--k')
ax[0].set_xlim(np.hstack(beta).max(), np.hstack(beta).min())
ax[0].set_xlabel("$\\beta$", fontsize=14)
ax[0].set_ylabel("$\phi_d$", fontsize=14)
if i_target < len(phi_d):
            ax[0].plot(beta[i_target], phi_d[i_target], "k*", ms=10, label=f"iter {i_target}")
            ax[0].legend(loc='best')

ax[1].semilogx(beta, phi_m)
ax[1].set_xlim(np.hstack(beta).max(), np.hstack(beta).min())
ax[1].set_xlabel("$\\beta$", fontsize=14)
ax[1].set_ylabel("$\phi_m$", fontsize=14)
if i_target < len(phi_d):
            ax[1].plot(beta[i_target], phi_m[i_target], "k*", ms=10)

ax[2].plot(phi_m, phi_d)
ax[2].set_xlim(np.hstack(phi_m).min(), np.hstack(phi_m).max())
ax[2].set_xlabel("$\phi_m$", fontsize=14)
ax[2].set_ylabel("$\phi_d$", fontsize=14)
if i_target < len(phi_d):
        ax[2].plot(phi_m[i_target], phi_d[i_target], "k*", ms=10)

plt.tight_layout()
plt.show()

ii=16
fname = "/home/cdut5815/下载/Simpeg/123/inv_data16.obs"
np.savetxt(fname, pred[ii],header='inv-1')
fname = "/home/cdut5815/下载/Simpeg/123/inv_model16.obs"
np.savetxt(fname,np.c_[mx,my,mz,model1[ii]], fmt="%.4f",header='model-1')
ii=20
fname = "/home/cdut5815/下载/Simpeg/123/inv_model20.obs"
np.savetxt(fname, pred[ii],header='inv-1')
fname = "/home/cdut5815/下载/Simpeg/123/inv_model20.obs"
np.savetxt(fname,np.c_[mx,my,mz,model1[ii]], fmt="%.4f",header='model-1')

n = "a1";

fig, ax = plt.subplots(1,3, figsize = (12,10))
clim = [10, 100]
dat1 = mesh.plotSlice(mrec, grid=False, ax=ax[0], ind=indy, normal='Y', cmap='RdBu_r')
dat2 = mesh.plotSlice(mrec, grid=False, ax=ax[1], ind=indx, normal='X',cmap='RdBu_r')
dat3 = mesh.plotSlice(mrec, grid=False, ax=ax[2], ind=indz,cmap='RdBu_r')
ax[0].set_xlim(xlim)
ax[0].set_ylim(zlim)
ax[1].set_xlim(ylim)
ax[1].set_ylim(zlim)
ax[2].set_xlim(xlim)
ax[2].set_ylim(ylim)
plt.tight_layout()
ax[0].set_aspect(1)
ax[1].set_aspect(1)
ax[2].set_aspect(1)
ax[0].set_title("Y={:1.0f}m".format(mesh.cell_centers_y[indy]))
ax[1].set_title("X={:=1.0f}m".format(mesh.cell_centers_x[indx]))
ax[2].set_title("Z={:1.0f}m".format(mesh.cell_centers_y[indz]))
#cb.set_label("$\sigma$ (S/m)")
plt.savefig('/home/cdut5815/下载/Simpeg/123/'+n+'.png')
plt.show()

