import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
import os

from Module_Parameters_3D import *


directory = 'data/rcs'



plots = 'plots/plots_RCS/cartesian'
if not os.path.exists(plots):
    os.makedirs(plots)


dtheta = 1
theta = np.arange(0, 360 + 1, dtheta)



BRCS_LBM   = np.load(directory+'/RCS_LBM_{}_{}_{}.npy'.format(ratio, er2, phi0))
BRCS_exact = np.load(directory+'/BRCS_exact_{}_{}_{}.npy'.format(ratio, er2, phi0))
    




###################################


plt.clf()

plt.rc('font', family = 'serif', size = 10)
plt.rc('xtick', labelsize = 10)
plt.rc('ytick', labelsize = 10)
plt.rc('lines', markersize = 2, lw = 0.75)
plt.rc('text', usetex = True)

################################################################################
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (3.35, 2.05), dpi=600, constrained_layout = True)




plt.plot(theta, BRCS_exact, 'k-')

plt.plot(theta, BRCS_LBM, 'r--')


plt.xticks([0, 180, 360])
plt.gca().set_xticklabels([0, r'$\pi$', r'$2\pi$'])

plt.xlabel(r'$\phi$')
plt.ylabel(r'$\sigma / (\pi a^2)$')


plt.legend([r'Analytical', r'LBM'])

##plt.grid(which='both')


plt.yscale('log')

    
plt.savefig(plots+'/RCS_{}_{}.svg'.format(ratio, er2, phi0))
plt.close()
################################################################################




################################################################################
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (3.35, 2.05), dpi=600, constrained_layout = True)




plt.plot(theta, np.absolute((BRCS_exact - BRCS_LBM) / BRCS_exact)*100, 'k-')

##plt.plot(phi, BRCS_LBM, 'r--')


plt.xticks([0, 180, 360])
plt.gca().set_xticklabels([0, r'$\pi$', r'$2\pi$'])

plt.xlabel(r'$\phi$')
plt.ylabel(r'$\sigma / (\pi a^2)$')

plt.grid(which='both')


plt.yscale('log')

    
plt.savefig(plots+'/RCS_{}_{}_Err.svg'.format(ratio, er2, phi0))
plt.close()
################################################################################

print(np.shape(BRCS_LBM))
print(np.shape(BRCS_exact))
print(np.shape(np.absolute((BRCS_exact - BRCS_LBM) / BRCS_exact)*100))



