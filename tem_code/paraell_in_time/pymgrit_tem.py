
from discretize.utils import make_property_tensor
from pymgrit.core.split import split_communicator
from pymgrit import *

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz, active_from_xyz
import numpy as np
from mpi4py import MPI
from pymgrit.core.application import Application
from pymgrit.core.vector import Vector
from pymgrit.core.mgrit import Mgrit
from SimPEG.data import SyntheticData, Data


from SimPEG.electromagnetics import time_domain as TDEM
from SimPEG import maps, data, simulation
from SimPEG.utils import plot2Ddata
from SimPEG import maps
import SimPEG.electromagnetics.time_domain as tdem
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

save_file = False
from pymgrit.core.vector import Vector

class Vectortdem(Vector):
    """
    Vector class for the Dahlquist test equation
    """
    def __init__(self, nx,ny):

        super().__init__()
        self.nx = nx #nx is the dimensions of the data per reciver point
        self.ny = ny # ny is num of recievers
        self.value = np.zeros((self.nx,self.ny))

    def __add__(self, other):

        tmp = Vectortdem(self.nx,self.ny)
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        tmp = Vectortdem(self.nx,self.ny)
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def __mul__(self, other):
        tmp = Vectortdem(self.nx,self.ny)
        tmp.set_values(self.get_values() * other)
        return tmp

    def norm(self):
        return np.linalg.norm(self.value)

    def clone(self):
        tmp = Vectortdem(self.nx,self.ny)
        tmp.value = self.value
        return tmp


    def clone_zero(self):
        tmp = Vectortdem(self.nx,self.ny)
        tmp.value = np.zeros((self.nx,self.ny))
        return tmp
    
    def clone_rand(self):

        tmp = Vectortdem(self.nx,self.ny)
        tmp.set_values(np.random.rand(self.nx,self.ny))
        return tmp

    def set_values(self, value):
        self.value = value

    def get_values(self):
        return self.value

    def pack(self):
        return self.value

    def unpack(self, value):
        self.value = value

class SimpegTDEM(Application):
    def __init__(self, simulation, survey, mesh,model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation = simulation
        self.survey = survey
        self.mesh = mesh
        self.model = model
        #nd = self.simulation.survey.nD #num of data
        nr = self.simulation.survey.nSrc
        nm = simulation.Me.size
        self.u = Vectortdem(nm,nr)
        self.vector_template = Vectortdem(nm,nr)
        self.vector_t_start = Vectortdem(nm,nr)

    def step(self,u_start,t_start, t_stop):
        # Define the time step size
        dt = t_stop - t_start
        self.simulation.time_steps = [(dt,1)]
        
        if t_start == self.t[0]:
            f = self.simulation.fields(model)
        else:
            f = self.simulation.fields(model,InitialFields = u_start.value)

        self.ini = f[:, simulation._fieldType + "Solution", 0]

        u_sol = f[:, simulation._fieldType + "Solution", 1]

        self.u.set_values(u_sol)

        return self.u


xx, yy = np.meshgrid(np.linspace(-3000, 3000, 101), np.linspace(-3000, 3000, 101))
zz = np.zeros(np.shape(xx))
topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

waveform_times = np.linspace(-0.002, 0, 21)

waveform = tdem.sources.TrapezoidWaveform(
    ramp_on=np.r_[-0.002, -0.001], ramp_off=np.r_[-0.001, 0.0], off_time=0.0
)

waveform_value = [waveform.eval(t) for t in waveform_times]

n_times = 3
time_channels = np.logspace(-4, -3, n_times)
# Defining transmitter locations
n_tx = 11
xtx, ytx, ztx = np.meshgrid(
    np.linspace(-200, 200, n_tx), np.linspace(-200, 200, n_tx), [50]
)
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
xrx, yrx, zrx = np.meshgrid(
    np.linspace(-200, 200, n_tx), np.linspace(-190, 190, n_tx), [30]
)
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

source_list = []  # Create empty list to store sources

# Each unique location defines a new transmitter
for ii in range(ntx):
    # Here we define receivers that measure the h-field in A/m
    dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
        receiver_locations[ii, :], time_channels, "z"
    )
    receivers_list = [
        dbzdt_receiver
    ]  # Make a list containing all receivers even if just one

    # Must define the transmitter properties and associated receivers
    source_list.append(
        tdem.sources.MagDipole(
            receivers_list,
            location=source_locations[ii],
            waveform=waveform,
            moment=1.0,
            orientation="z",
        )
    )

survey = tdem.Survey(source_list)

dh = 25.0  # base cell width
dom_width = 1600.0  # domain width
nbc = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))  # num. base cells

# Define the base mesh
h = [(dh, nbc)]
mesh = TreeMesh([h, h, h], x0="CCC")

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, topo_xyz, octree_levels=[0, 0, 0, 1], method="surface", finalize=False
)

# Mesh refinement near transmitters and receivers
mesh = refine_tree_xyz(
    mesh, receiver_locations, octree_levels=[2, 4], method="radial", finalize=False
)

# Refine core mesh region
xp, yp, zp = np.meshgrid([-250.0, 250.0], [-250.0, 250.0], [-250.0, 0.0])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 4], method="box", finalize=False)

mesh.finalize()


air_conductivity = 1e-8
background_conductivity = 2e-3
block_conductivity = 2e0

# Active cells are cells below the surface.
ind_active = active_from_xyz(mesh, topo_xyz)
model_map = maps.InjectActiveCells(mesh, ind_active, air_conductivity)

# Define the model
model = background_conductivity * np.ones((ind_active.sum()))
ind_block = (
    (mesh.gridCC[ind_active, 0] < 100.0)
    & (mesh.gridCC[ind_active, 0] > -100.0)
    & (mesh.gridCC[ind_active, 1] < 100.0)
    & (mesh.gridCC[ind_active, 1] > -100.0)
    & (mesh.gridCC[ind_active, 2] > -200.0)
    & (mesh.gridCC[ind_active, 2] < -50.0)
)

model[ind_block] = block_conductivity

# Plot log-conductivity model
mpl.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(7, 6))

log_model = np.log10(model)

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

time_steps = [(1e-4, 20), (1e-5, 10), (1e-4, 10)]

simulation = tdem.simulation.Simulation3DElectricField(
    mesh, survey=survey, sigmaMap=model_map, solver=Solver, t0=-0.002
)

model_w = background_conductivity * np.ones((mesh.nC))
simulation.model = model

b = len(simulation.mesh)
a = simulation.MeSigmaI
M_isotropic = make_property_tensor(mesh, model_w)

# Set the time-stepping for the simulation
simulation.time_steps = time_steps
for i in range(len(time_steps)):
    a = time_steps[i]

    if i == 0:
        tmp = np.linspace(0, a[0]*a[1], a[1])
        end_p = tmp[-1]
        t = tmp
    else:
        tmp = np.linspace(end_p+a[0], a[0]*a[1]+end_p, a[1])
        end_p = tmp[-1]
        t = np.hstack((t,tmp))

t_interval = t


comm_world = MPI.COMM_WORLD
comm_x, comm_t = split_communicator(comm_world, 1)
problem = SimpegTDEM(simulation, survey, mesh,model,t_interval = t_interval)


dahlquist_multilevel_structure = simple_setup_problem(problem=problem, level=2, coarsening=2)
# Define the MGRIT solver
mgrit = Mgrit(
    dahlquist_multilevel_structure,
    tol=1e-15,
    max_iter=50,comm_time=comm_t, comm_space=comm_x
)

# Solve the problem
mgrit.solve()



