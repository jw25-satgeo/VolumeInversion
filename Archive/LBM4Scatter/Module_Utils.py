import numpy as np




'''TMz plane wave'''
def planeWaveTM(Ez, Hy, t, omega, x, ymin, ymax, zmin, zmax):
    Ez[zmin:zmax, ymin:ymax, x] =  np.sin(omega * t)
    Hy[zmin:zmax, ymin:ymax, x] = - np.sin(omega * t)


'''TEz plane wave'''
def planeWaveTE(Ey, Hz, t, omega, x, ymin, ymax, zmin, zmax):
    Ey[zmin:zmax, ymin:ymax, x] = np.sin(omega * t)
    Hz[zmin:zmax, ymin:ymax, x] = np.sin(omega * t)


import sys
import time
import numpy as np
from Module_Geometry_3D import *



'''number of parallel threads'''
N = 4


'''Accessing command line arguments'''
parameters = sys.argv

theta = 45
######################

a, ratio = 25, 0.5   # ratio = a / wavelength
n = 4
Nx, Ny, Nz = n*a, n*a, n*a  # size of the computational domain

er1, mur1, er2, er3 = 1, 1, 36, 10000   # material properties i.e. permittivity and permeabilty


################################################################
###                        SCATTERER                        ####
################################################################

'''initilizing the polar coordinates'''
def initialize_field(Nz=10, Ny=10, Nx=10):
    return np.zeros((Nz, Ny, Nx), dtype=np.float32, order='C')
r, theta, phi = [initialize_field(Nz, Ny, Nx) for _ in range(3)]


'''initilizing the domain properties'''
def initialize_material_properties(er1=1, mur1=1, Nz=10, Ny=10, Nx=10):
    yield np.ones((Nz, Ny, Nx), dtype=np.float32, order='C') * er1
    yield np.ones((Nz, Ny, Nx), dtype=np.float32, order='C') * mur1
    return

er, mur = initialize_material_properties(er1, mur1, Nz, Ny, Nx)

'''center of the scatterer'''
cx = Nx//2 + 0.5
cy = Ny//2 + 0.5
cz = Nz//2 + 0.5

'''converting from cartesian to spherical coordinates'''
r, theta, phi = carToSpherical(Nz, Ny, Nx, cz, cy, cx)

'''scatterer particle'''
sphere(r, er, er2, a)



################################################################


######################
# LBM properties (DO NOT CHANGE)
Q = 7               # number of velocities at a grid
velocity = 1.0/3    # velocity of EM wave in vacuum
######################


######### DON'T CHANGE ########
wavelength = a / ratio
omega = 2 * np.pi * velocity / wavelength
period = int(np.round(wavelength / velocity))

'''wavenumber of the incident wave'''
k = 2*np.pi / wavelength
###############################


#############################
# boundary of EM wave source
xloc = 0
ymin = 0
ymax = Ny
zmin = 0
zmax = Nz
#############################



'''half-width of the square bounding box'''
w = int(np.round(1.5*a))

'''box surrounding the scatterer'''
Top    = np.arange(int(cz - w), int(cz + w)), int(cy + w), np.arange(int(cx - w), int(cx + w))
Bottom = np.arange(int(cz - w), int(cz + w)), int(cy - w), np.arange(int(cx - w), int(cx + w))
Right  = np.arange(int(cz - w), int(cz + w)), np.arange(int(cy - w), int(cy + w)), int(cx + w)
Left   = np.arange(int(cz - w), int(cz + w)), np.arange(int(cy - w), int(cy + w)), int(cx - w)
Front  = int(cz + w), np.arange(int(cy - w), int(cy + w)), np.arange(int(cx - w), int(cx + w))
Back   = int(cz - w), np.arange(int(cy - w), int(cy + w)), np.arange(int(cx - w), int(cx + w))

'''unit normal vectors at the perimeter of the bounding box'''
nxTop, nxBottom, nxRight, nxLeft, nxFront, nxBack = 0, 0, 1, -1, 0, 0
nyTop, nyBottom, nyRight, nyLeft, nyFront, nyBack = 1, -1, 0, 0, 0, 0
nzTop, nzBottom, nzRight, nzLeft, nzFront, nzBack = 0, 0, 0, 0, 1, -1







noOfPeriods = 10
noOfReflections = 0


'''number of time steps the code will run'''
Time = int(3 * (Nx * np.sqrt(er1) + noOfReflections * 2 * a * np.sqrt(er2)) + noOfPeriods * period)




import os
from ctypes import CDLL, POINTER
from ctypes import c_size_t, c_double, c_float, c_int
import numpy as np


###############################################################################################################
########                                           SHARED LIBRARY                                   ###########
###############################################################################################################

'''loading the shared file (c library)'''
path = os.getcwd()
myclib = CDLL(os.path.join(path, "LBM_3D.so"))

'''defining 2D and 3D pointers (LBM runs in C, for that pointer is needed)'''
P3D = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags="C")
P4D = np.ctypeslib.ndpointer(dtype=np.float32, ndim=4, flags="C")

'''calculation of macroscopic fields (FUNCTION PROTOTYPE)'''
myclib.macroField.argtypes = [P4D, P3D, P3D, c_int, c_int, c_int, c_int, c_int]
myclib.macroField.restype  = None

'''initilization of macroscopic fields (FUNCTION PROTOTYPE)'''
myclib.initializeField.argtypes = [P3D, P3D, P3D, P3D, P3D, P3D, c_int, c_int, c_int, c_int]
myclib.initializeField.restype  = None

'''collision and streaming (FUNCTION PROTOTYPE)'''
myclib.collNotForcingNode.argtypes = [P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P3D, P3D, P3D, P3D, P3D, P3D, P3D, P3D, c_int, c_int, c_int, c_int, c_int]
myclib.collNotForcingNode.restype  = None


myclib.collForcingNode.argtypes = [P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P3D, P3D, P3D, P3D, P3D, P3D, P3D, P3D, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
myclib.collForcingNode.restype  = None

myclib.streaming.argtypes = [P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, P4D, c_int, c_int, c_int, c_int, c_int]
myclib.streaming.restype  = None

###############################################################################################################


import numpy as np


############################################# INSTANTANEOUS TRACTION FORCE ######################################

'''instantaneous traction vector in x direction'''
def fxInstantaneous(Ex, Ey, Ez, Hx, Hy, Hz, er1, mur1, nx, ny, nz):

    Txx = er1 * (Ex*Ex - 0.5 * (Ex*Ex + Ey*Ey + Ez*Ez)) + mur1 * (Hx*Hx - 0.5 * (Hx*Hx + Hy*Hy + Hz*Hz))
    Txy = er1 * (Ex*Ey) + mur1 * (Hx*Hy)
    Txz = er1 * (Ex*Ez) + mur1 * (Hx*Hz)

    fx = Txx * nx + Txy * ny + Txz * nz

    return fx



'''instantaneous traction vector in y direction'''
def fyInstantaneous(Ex, Ey, Ez, Hx, Hy, Hz, er1, mur1, nx, ny, nz):

    Tyx = er1 * (Ey*Ex) + mur1 * (Hy*Hx)
    Tyy = er1 * (Ey*Ey - 0.5 * (Ex*Ex + Ey*Ey + Ez*Ez)) + mur1 * (Hy*Hy - 0.5 * (Hx*Hx + Hy*Hy + Hz*Hz))
    Tyz = er1 * (Ey*Ez) + mur1 * (Hy*Hz)

    fy = Tyx * nx + Tyy * ny + Tyz * nz

    return fy



'''instantaneous traction vector in z direction'''
def fzInstantaneous(Ex, Ey, Ez, Hx, Hy, Hz, er1, mur1, nx, ny, nz):

    Tzx = er1 * (Ez*Ex) + mur1 * (Hz*Hx)
    Tzy = er1 * (Ez*Ey) + mur1 * (Hz*Hy)
    Tzz = er1 * (Ez*Ez - 0.5 * (Ex*Ex + Ey*Ey + Ez*Ez)) + mur1 * (Hz*Hz - 0.5 * (Hx*Hx + Hy*Hy + Hz*Hz))

    fz = Tzx * nx + Tzy * ny + Tzz * nz

    return fz

################################################################################################################




############################################# AVERAGE TRACTION FORCE ###########################################

'''average (over one time period) traction vector in x direction'''
def fxAverage(Ex, Ey, Ez, Hx, Hy, Hz, er1, mur1, nx, ny, nz):

    Ex_, Ey_, Ez_ = np.conjugate(Ex), np.conjugate(Ey), np.conjugate(Ez)
    Hx_, Hy_, Hz_ = np.conjugate(Hx), np.conjugate(Hy), np.conjugate(Hz)

    Txx = 0.5 * np.real(er1 * (Ex*Ex_ - 0.5 * (Ex*Ex_ + Ey*Ey_ + Ez*Ez_)) + mur1 * (Hx*Hx_ - 0.5 * (Hx*Hx_ + Hy*Hy_ + Hz*Hz_)))
    Txy = 0.5 * np.real(er1 * (Ex*Ey_) + mur1 * (Hx*Hy_))
    Txz = 0.5 * np.real(er1 * (Ex*Ez_) + mur1 * (Hx*Hz_))

    fx = Txx * nx + Txy * ny + Txz * nz

    return fx



'''average (over one time period) traction vector in y direction'''
def fyAverage(Ex, Ey, Ez, Hx, Hy, Hz, er1, mur1, nx, ny, nz):

    Ex_, Ey_, Ez_ = np.conjugate(Ex), np.conjugate(Ey), np.conjugate(Ez)
    Hx_, Hy_, Hz_ = np.conjugate(Hx), np.conjugate(Hy), np.conjugate(Hz)

    Tyx = 0.5 * np.real(er1 * (Ey*Ex_) + mur1 * (Hy*Hx_))
    Tyy = 0.5 * np.real(er1 * (Ey*Ey_ - 0.5 * (Ex*Ex_ + Ey*Ey_ + Ez*Ez_)) + mur1 * (Hy*Hy_ - 0.5 * (Hx*Hx_ + Hy*Hy_ + Hz*Hz_)))
    Tyz = 0.5 * np.real(er1 * (Ey*Ez_) + mur1 * (Hy*Hz_))

    fy = Tyx * nx + Tyy * ny + Tyz * nz

    return fy



'''average (over one time period) traction vector in z direction'''
def fzAverage(Ex, Ey, Ez, Hx, Hy, Hz, er1, mur1, nx, ny, nz):

    Ex_, Ey_, Ez_ = np.conjugate(Ex), np.conjugate(Ey), np.conjugate(Ez)
    Hx_, Hy_, Hz_ = np.conjugate(Hx), np.conjugate(Hy), np.conjugate(Hz)

    Tzx = 0.5 * np.real(er1 * (Ez*Ex_) + mur1 * (Hz*Hx_))
    Tzy = 0.5 * np.real(er1 * (Ez*Ey_) + mur1 * (Hz*Hy_))
    Tzz = 0.5 * np.real(er1 * (Ez*Ez_ - 0.5 * (Ex*Ex_ + Ey*Ey_ + Ez*Ez_)) + mur1 * (Hz*Hz_ - 0.5 * (Hx*Hx_ + Hy*Hy_ + Hz*Hz_)))

    fz = Tzx * nx + Tzy * ny + Tzz * nz

    return fz


#################################################################################################################
