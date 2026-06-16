import numpy as np




'''TMz plane wave'''
def planeWaveTM(Ex, Hy, t, omega, z, ymin, ymax, xmin, xmax):
    Ex[z, ymin:ymax, xmin:xmax] =  np.sin(omega * t)
    Hy[z, ymin:ymax, xmin:xmax] =  np.sin(omega * t)


'''TEz plane wave'''
def planeWaveTE(Ey, Hz, t, omega, x, ymin, ymax, zmin, zmax):
    Ey[zmin:zmax, ymin:ymax, x] = np.sin(omega * t)
    Hz[zmin:zmax, ymin:ymax, x] = np.sin(omega * t)


