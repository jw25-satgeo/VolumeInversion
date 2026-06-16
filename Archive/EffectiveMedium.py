# REFERENCE : Forward Electromagnetic Scattering Models for Sea Ice - 
# (Golden et al., 1998) @ https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=718637

import numpy as np
import scipy
import math
import cmath

from PhreeqcSolver import frezchem_seawater

def solve_bruggeman(T, S, wavelength=0.229, orientation=[0,0,1], shp_ratio=1, ice_solver=None):
    '''
    T: temperature in Celsius
    S: salinity in psu
    wavelength: wavelength in metres
    orientation: list of 3 floats for brine pocket orientation
    shp_ratio: L/D ratio along principal axis of ellipsoidal brine pocket
    '''
    if (ice_solver is None): ice_solver = frezchem_seawater()
    brine_frac = 1 - ice_solver.solve_at(T,S)[0]
    ice_rdie = (3.15 + 0.01j) * np.eye(3,dtype=np.complex128)
    bri_rdie = GW2020_pebrine(T, S, wavelength) * np.eye(3,dtype=np.complex128)
    L = initialise_ellipsoid_depolarisation_matrix(orientation, shp_ratio)
    size = 1e-2 #assume particle diameter is ~1/100 x wavelength

    eff = maxwell_sihvola(ice_rdie, bri_rdie, L, brine_frac, size)
    eff = bruggeman_iter(ice_rdie, bri_rdie, L, brine_frac, size, eff)
    
    return eff

c = 299792458 #m/s (SI)
pe_0 = 8.8541878188e-12 #vacuum permittivity F/m
def GW2020_pebrine(T, S, wavelength):
    '''T: temperature, Celsius \nS: salinity, psu'''
    #!IMPLEMENTED ------------------------------------------------------------------------------------------------------------------------- 
    # Seawater Debye Model Function at L-band and Its Impact on Salinity Retrieval from Aquarius Satellite Data .. Zhou et al. 2021
    # - https://ntrs.nasa.gov/api/citations/20210016626/downloads/TGRS-2020-01490.R1_Proof_seawater.pdf
    # - https://ieeexplore.ieee.org/document/9325088 (better source)
    #   e_dw = e_inf + (e_{s-dw}(T) - e_inf)/(1+jwt(T)) || for pure water
    #   e_inf = 4.9
    #   e_{s-dw} : static dielectric constant
    #   t(T) : relaxation time
    # .. e_{s-dw} = 8.80516 - 4.01796e-1*T - 5.10271e-5*T**2 + 2.55892E-05*T**3
    # .. t(T) = 1.75030e-11 - 6.12993e-13*T + 1.24504e-14*T**2 - 1.14927E-16*T**3
    # e_sw(S,T) = e_inf + (e_{s-dw}(T)*R_{sw-dw}(S,T) - e_inf)/(1+jwt(T) - (j*s(S,T))/we_0     || for salt water
    # .. R_{sw-dw}(S,T) = 1 - S*(3.97185E-03 - 2.49205E-05*T - 4.27558E-05*S +  3.92825E-07*ST + 4.15350E-07*S**2)  || ionic term
    # .. s(S,T)   || conductivity of seawater
    # .. s(S,0) = 9.50470e-2 * S - 4.30858e-4 * S**2 + 2.16182e-6 * S**3
    # .. s(S,T) = s(S,0) * R_s(S,T)  || R_s(S,T) = 1+T*(3.76017e-2+6.32830e-5*T + 4.83420e-7*T**2 - 3.97484e-4*S + 6.26522e-6*S**2)
    #
    #  ..from which can be extracted the brine dielectric constant. Yay! (pain, it's all empirical and the measurements are limited in range)
    #
    afreq = 2 * math.pi * c / wavelength
    swmdw_ionic_R = 1 - S*(3.97185E-03 - 2.49205E-05*T - 4.27558E-05*S +  3.92825E-07*S*T + 4.15350E-07*S**2)
    sw_ionic_R = 1+T*(3.76017e-2+6.32830e-5*T + 4.83420e-7*T**2 - 3.97484e-4*S + 6.26522e-6*S**2)
    sw_conductivityS0 = 9.50470e-2 * S - 4.30858e-4 * S**2 + 2.16182e-6 * S**3
    sw_conductivity = sw_conductivityS0 * sw_ionic_R
    sw_relaxation_time = 1.75030e-11 - 6.12993e-13*T + 1.24504e-14*T**2 - 1.14927E-16*T**3
    e_smdw = 8.80516e+1 - 4.01796e-1*T - 5.10271e-5*T**2 + 2.55892E-05*T**3
    pe_brine = 4.9 + ((e_smdw*swmdw_ionic_R - 4.9) / (1+1j*afreq*sw_relaxation_time)) - (1j*sw_conductivity / (afreq*pe_0))
    return pe_brine

# F R O M :
#!/usr/bin/python
#
# Copyright 2024 John Kendrick & Andrew Burnett
#
# This file is part of PDielec  @  https://github.com/JohnKendrick/PDielec/blob/master/PDielec/Calculator.py
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# You should have received a copy of the MIT License along with this program, if not see https://opensource.org/licenses/MIT
#

def calculate_size_factor (x):
    """Calculate a size effect using Equations 10.38 and 10.39 in Sihvola."""
    if x < 1.0E-12:
        result = 1.0
    else:
        ix = complex(0,x)
        g1 = 2.0 / 3.0 * ( ( 1.0 + ix ) * np.exp(-ix) - 1.0 )
        g2 = ( 1 + ix - (7.0/15.0) * x*x - complex(0,2.0*x*x*x/15.0) ) * np.exp(-ix) -1.0
        result = 1 - g1 - g2
    return result

def initialise_ellipsoid_depolarisation_matrix(unique, aoverb):
    """Initialise a 3x3 tensor with the ellipsoid depolarisation matrix.

    Parameters
    ----------
    unique : list of 3 floats
        Unique direction for ellipsoid
    aoverb : float
        The ratio of a / b (the ratio of the principle axis lengths of the ellipsoid)

    Returns
    -------
    np.array
        The Ellipsoid dpolarisation tensor
    """
    # unique is the unique direction of the ellipsoid
    # a and b are the sizes of the ellipsoid along the unique axis (a) and perpendicular to it b
    # if a > b then we have a prolate ellipsoid
    unique = unique / np.linalg.norm(unique)
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    xdotn, ydotn, zdotn = np.dot(x, unique), np.dot(y, unique), np.dot(z, unique)
    # choose the direction with the smallest projection along the ellipsoid
    # then project out any ellipsoid direction
    if np.abs(xdotn) <= np.abs(ydotn) and np.abs(xdotn) <= np.abs(zdotn): dir1 = x - (xdotn * unique)
    elif np.abs(ydotn) <= np.abs(xdotn) and np.abs(ydotn) <= np.abs(zdotn): dir1 = y - (ydotn * unique)
    else: dir1 = z - (zdotn * unique)
    dir1 = dir1/np.linalg.norm(dir1)
    # now find the orthogonal direction
    dir2 = np.cross(unique, dir1)
    dir2 = dir2/np.linalg.norm(dir2)
    bovera = 1.0 / aoverb
    small = 1.0E-8
    if bovera < 1.0-small:
        e = math.sqrt(1.0  - bovera*bovera)
        nz = (1 - e*e) * (np.log((1+e) / (1-e)) - 2*e) / (2*e*e*e)
    elif bovera > 1.0+small:
        e = math.sqrt(bovera*bovera - 1.0)
        nz = (1 + e*e) * (e - np.arctan(e)) / (e*e*e)
    else: nz = 1.0/3.0
    nxy = (1 - nz) * 0.5
    # compute the tensors from the outer product of each direction
    tensor = nz*np.outer(unique, unique) + nxy*np.outer(dir1, dir1) + nxy*np.outer(dir2, dir2)
    return tensor / np.trace(tensor)

def bruggeman_iter( eps1, eps2, L, f2, size, epsbr):
    """Calculate an iteration of the Bruggeman method..

    Parameters
    ----------
    eps1 : array_like (3x3)
        The dielectric constant tensor of medium 1.
    eps2 : array_like (3x3)
        The dielectric constant tensor of medium 2.
    shape : str
        The name of the current shape.
    L : array_like (3x3)
        The shape's depolarisation matrix.
    f2 : float
        The volume fraction of component 2.
    size : float
        The dimensionless size parameter for the frequency under consideration.
    epsbr : float (3x3)
        An initial guess at the solution.

    Returns
    -------
    tensor (3x3)
        The effective dielectric constant.

    Notes
    -----
    This function applies homogenization formalisms to active dielectric composite materials as discussed in the work of Tom G. Mackay and Akhlesh Lakhtakia.

    """
    f1 = 1.0 - f2
    # perform an iteration
    converged = False
    niters = 0
    while not converged:
        niters += 1
        epsbr, error = _brug_iter_error(epsbr, eps1, eps2, L, f1, size)
        if abs(error) < 1.0E-8:
            converged = True
        if niters > 3000:
            print("Bruggeman iterations failed, error=", error)
            converged = True
    return epsbr

def _brug_iter_error(epsbr, eps1, eps2, L, f1, size):
    f2 = 1.0 - f1
    # If appropriate calculate a size effect using Equations 10.38 and 10.39 in Sihvola
    size_factor = calculate_size_factor(size)
    leps1 = np.dot(L, (eps1 - epsbr))
    leps2 = size_factor * np.dot(L, (eps2 - epsbr))
    a1 = np.linalg.inv(epsbr + leps1)
    a2 = np.linalg.inv(epsbr + leps2)
    # alpha1 and 2 are the polarisabilities of 1 and 2 in the effective medium
    alpha1 = np.dot((eps1 - epsbr), a1)
    alpha2 = np.dot((eps2 - epsbr), a2)
    # the error or residual matrix should be zero for a bruggeman solution
    error = f1*alpha1 + f2*alpha2
    error = np.linalg.norm(error)
    m1 = f1*np.dot(eps1, a1)+f2*np.dot(eps2, a2)
    m2 = np.linalg.inv(f1*a1 + f2*a2)
    damp = 0.0
    epsbr = (1.0 - damp)*np.dot(m1, m2) + damp*epsbr
    return epsbr, error

def maxwell_sihvola(eps1, eps2, L, vf, size):
    """Calculate the effective constant permittivity using the Maxwell Garnett method.

    Parameters
    ----------
    eps1 : tensor (3x3)
        The dielectric constant tensor of the medium.
    eps2 : tensor (3x3)
        The total frequency dielectric constant tensor at the current frequency.
    L : matrix (3x3)
        The inclusion shape's depolarisation matrix.
    size : float
        The dimensionless size parameter for the frequency under consideration.
    vf : float
        The volume fraction of filler.

    Returns
    -------
    tensor
        The effective dielectric constant.

    """
    unit = np.eye(3)
    # Equation 6.29 on page 123 of Sihvola
    # Equation 6.40 gives the averaging over the orientation function
    # See also equation 5.80 on page 102 and equation 4.31 on page 70
    # If appropriate calculate a size effect using Equations 10.38 and 10.39 in Sihvola
    size_factor = calculate_size_factor(size)
    # calculate the polarisability matrix x the number density of inclusions
    nA = vf*np.dot((eps2-eps1), np.linalg.inv(unit + (size_factor * eps1 * np.dot(L, (eps2 - eps1)))))
    nAL = np.dot((nA), L)
    nA = np.trace(nA) / 3.0 * unit # average the polarisability over orientation
    nAL = np.trace(nAL) / 3.0 * eps1 * unit # average the polarisability*L over orientation
    # Calculate the average polarisation factor which scales the average field
    # based on equation 5.80 : <P> = pol . <E>
    pol = np.dot(np.linalg.inv(unit - nAL), nA)
    # Meff . <E> = Me . <E> + <P>
    # Meff . <E> = Me. <E> + pol . <E>
    # Meff = Me + pol
    effd         = eps1 + pol
    # Average over orientation
    trace = np.trace(effd) / 3.0
    return np.array([[trace, 0, 0], [0, trace, 0], [0, 0, trace]])


    
def Bruggeman_Spherical(nM, kM, nI, kI, VF):
    """
    Bruggemann function.
    Allow us to calculte the complex refractive index of a mixture of two materials, using an EMA (Effective Medium Approximation)
    
    Parameters
    ----------
    nM : array
    Real part of refractive index of host Matrix (M is for Matrix)
    \nkM : array
    Complex part of refractive index of host Matrix (M is for Matrix)
    \nnI : array
    Real part of refractive index of inclusion (I is for Inclusion)
    \nkI : TYPE
    Complex part of refractive index of inclusion (I is for Inclusion)
    \nVF : int
    Volumic Fraction of inclusions in host matrix. Number between 0 et 1 (0 and 100%)     
    
    Returns
    -------
    nEffective : array
    Real part of the refractive index of the effective medium : the mixture of the host Matrix and the embedded particules
    \nkEffective : array
    Complex part of the refractive index of the effective medium : the mixture of the host Matrix and the embedded particules
    
    Noted than If vf = 0 :
    \nnEffective = nM and kEffective = kM
    \nNoted than If vf = 1.0 : 
    \nnEffective = nI and kEffective = kI
    
    \nSOURCE - https://github.com/SolPOCandCo/SolPOC/blob/main/solpoc/functions_SolPOC.py#L609
    \nCITE:
    @PHDTHESIS{grosj2018,
    url = "http://www.theses.fr/2018PERP0002",
    title = "Etude, modélisation et optimisation de surfaces fonctionnelles pour les collecteurs solaires thermiques à concentration",
    author = "Grosjean, Antoine",
    year = "2018",
    note = "Thèse de doctorat dirigée par Thomas, Laurent et Soum-Glaude, Audrey Sciences pour l’ingénieur Perpignan 2018",
    note = "2018PERP0002",
    url = "http://www.theses.fr/2018PERP0002/document",
    }
    """
    if VF == 0:
        return nI, kI

    eM = (nM + 1j * kM) ** 2
    eI = (nI + 1j * kI) ** 2
    y = 2

    # Calculating coefficients for the quadratic equation
    a = -y
    b = (VF * y + VF - 1) * eI - (VF * y + VF - y) * eM
    c = eM * eI

    # Solving the quadratic equation using NumPy's vectorized roots
    discriminant = np.sqrt(b**2 - 4 * a * c)
    e1 = (-b + discriminant) / (2 * a)
    e2 = (-b - discriminant) / (2 * a)

    # Choosing the appropriate roots
    positive_imaginary = np.imag(e1) > 0
    positive_real = np.real(e1) > 0

    Neffective = np.where(positive_imaginary, np.sqrt(e1),
                          np.where(np.imag(e2) > 0, np.sqrt(e2),
                          np.where(positive_real, np.sqrt(e1), np.sqrt(e2))))

    # Extracting the real and imaginary parts
    nEffective = np.real(Neffective)
    kEffective = np.imag(Neffective)

    return nEffective, kEffective






# -- old --



#DEBYE DIELECTRIC MODEL FUNCTION FOR SEAWATER BASED ON EXPANDED L-BAND MEASUREMENT DATA SET
#.. (https://ntrs.nasa.gov/api/citations/20210016627/downloads/IGARSS2020_Debye.pdf) (???)
#  The Effect of Salt Concentration on Dielectric Permittivity and Interfacial Polarization in Carbonate Rocks with Complex Pore Structure 
# ..  https://onepetro.org/SJ/article/29/06/3017/543478/The-Effect-of-Salt-Concentration-on-Dielectric
#The results documented in this paper show that due to the combined influence of interfacial polarization, pore structure, and NaCl salt concentration, 
#the application of the conventional mixture models in complex carbonate rock samples for water saturation assessment might not be reliable at frequencies 
#below 5 GHz. This is an important observation because the conventional dielectric permittivity tools work only up to 1 GHz frequency''' 
# aaaaaaaaaa (why)
#
#Accurate measurements of the dielectric constant of seawater at L band - https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015RS005776
# taken from relevant figures : 34, 35, 36 @ 1.4 GHz, 0 deg.C
# @ 29.969 psu : 78.39 + 43.25
# @ 33.004 psu : 77.35 + 46.41
# @ 34.992 psu : 77.18 + 47.93
# @ 38.024 psu : 76.55 + 50.65
#

# Mixing Rules with Complex Dielectric Coefficients 
# Sihvola, 2000 (https://link.springer.com/article/10.1023/A:1026511515005)
#  found from : Anisotropy of seasonal snow measured by polarimetric phase differences in radar time series
#              Leinss et al. 2016 (https://tc.copernicus.org/articles/10/1771/2016/tc-10-1771-2016.pdf)
# 
#def MaxwellGarnettAligned_pemix(bvf, T, S, wavelength, a=[1,1,1], f_pebrine='GW2020'):
#    '''a : [a_x, a_y, a_z] structural anisotropy'''
#    pe_ice = 3.15 + 0.01j
#    if (f_pebrine == 'GW2020'):
#        pe_brine = GW2020_pebrine(T,S)
#    else: raise Exception('brine permittivity formula could not be found. Available formulas are : GW2020')
#
#    if (not isinstance(a, np.ndarray)):
#        a = np.array(a)
#    
#    #Depolarization factors N_i
#    if (a[0] > a[1]):
#        ecc = (1 - a[1]**2 / a[0]**2)**0.5 #eccentricity
#        if (a[0] > a[1] * 2):
#            N_1 = (1 - ecc**2)/(2*ecc**3) * (np.log((1+ecc)/(1-ecc)) - 2*ecc)
#            N_2 = N_3 = 0.5*(1 - N_1)
#        else:
#            N_1 = 0.3333 - (2/15) * ecc**2
#            N_2 = N_3 = 0.3333 + (1/15) * ecc**2
#    elif (a[1] > a[2]):
#        ecc = (a[0]**2 / a[2]**2 - 1)**0.5
#        if (a[1] > a[2] * 2):
#            N_3 = (1 - ecc**2)/(ecc**3) * (ecc - np.arctan(ecc))
#            N_1 = N_2 = 0.5*(1 - N_3)
#        else:
#            N_3 = 0.3333 + (2/15) * ecc**2
#            N_1 = N_2 = 0.3333 - (1/15) * ecc**2
#    else: raise ValueError("spheroids of non-uniaxial anisotropy do not have a closed form function.")
#    pe_mix_1 = pe_ice + bvf*pe_ice*((pe_brine - pe_ice) / (pe_ice + (1-bvf)*N_1*(pe_brine - pe_ice)))
#    pe_mix_2 = pe_ice + bvf*pe_ice*((pe_brine - pe_ice) / (pe_ice + (1-bvf)*N_2*(pe_brine - pe_ice)))
#    pe_mix_3 = pe_ice + bvf*pe_ice*((pe_brine - pe_ice) / (pe_ice + (1-bvf)*N_3*(pe_brine - pe_ice)))
#    return np.array((pe_mix_1, pe_mix_2, pe_mix_3))

# Maxwell-Garnett, Bruggeman formula reference - https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-33-7-1244&id=344272#e18

#def MaxwellGarnett_pemix(bvf, T, S, wavelength, f_pebrine='GW2020'):
#    '''bvf: brine volume fraction\nT: temperature, celsius\nS: salinity, psu\nwavelength: metres'''
#    k0 = 2*np.pi/wavelength
#    
#    #permittivities:
#    pe_ice = 3.15 + 0.01j #ice permittivity largely insensitive to frequency @ L-band and above
#
#    if (f_pebrine == 'GW2020'):
#        pe_brine = GW2020_pebrine(T,S, wavelength)
#    else: raise Exception('brine permittivity formula could not be found. Available formulas are : GW2020')
#
#    #spherical inclusions mixing
#    a = pe_brine + 2*pe_ice + 2*bvf*(pe_brine - pe_ice)
#    b = pe_brine + 2*pe_ice - bvf*(pe_brine - pe_ice)
#    pe_mix = pe_ice * a / b
#
#    return pe_mix
    
# An analysis of the tensor dielectric constant of sea ice at microwave frequencies
#   (Office of Naval Research Department of the Navy, Report 7975, October 1985)
# https://apps.dtic.mil/sti/tr/pdf/ADA168015.pdf
#def Navy1985_pemix(bvf, T, S, wavelength):
#    print("...later.")
#    pass

#def Stogryn1985_pebrine(T, S, wavelength):
#    print("sorry, no luck here!")
#    pass
    # The dielectric properties of brine in sea ice at microwave frequencies
    # Stogryn & Desargant 1985 (https://doi.org/10.1109/TAP.1985.1143610)
    #
    # "Unfortunately, all previously available descriptions of the dielectric constant of brine at the salinities and temperatures found in naturally
    #  occurring sea ice have been based on extra- polations of measurements made on NaCl solutions at temperatures above 0°C." <- too real, argh
    #
    # ... wait a second. There's no frequency samples at L-band.
    #      ...
    #      ..nooooooooooo whyyyyyyyyyyyyyyy