# REFERENCE : Forward Electromagnetic Scattering Models for Sea Ice - 
# (Golden et al., 1998) @ https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=718637

import numpy as np
import scipy

c = 299792458 #m/s (SI)
pe_0 = 8.8541878188e-12 #vacuum permittivity F/m

    #refr_ind = np.sqrt(pe_mix)
    #absorption = k0 * -refr_ind.imag

# Mixing Rules with Complex Dielectric Coefficients 
# Sihvola, 2000 (https://link.springer.com/article/10.1023/A:1026511515005)
#  found from : Anisotropy of seasonal snow measured by polarimetric phase differences in radar time series
#              Leinss et al. 2016 (https://tc.copernicus.org/articles/10/1771/2016/tc-10-1771-2016.pdf)
# 
def MaxwellGarnettAligned_pemix(bvf, T, S, wavelength, a=[1,1,1], f_pebrine='GW2020'):
    '''a : [a_x, a_y, a_z] structural anisotropy'''
    pe_ice = 3.15 + 0.01j
    if (f_pebrine == 'GW2020'):
        pe_brine = GW2020_pebrine(T,S)
    else: raise Exception('brine permittivity formula could not be found. Available formulas are : GW2020')

    if (!isinstance(a, np.ndarray)):
        a = np.array(a)
    
    #Depolarization factors N_i
    if (a[0] > a[1]):
        ecc = (1 - a[1]**2 / a[0]**2)**0.5 #eccentricity
        if (a[0] > a[1] * 2):
            N_1 = (1 - ecc**2)/(2*ecc**3) * (np.log((1+ecc)/(1-ecc)) - 2*ecc)
            N_2 = N_3 = 0.5*(1 - N_1)
        else:
            N_1 = 0.3333 - (2/15) * ecc**2
            N_2 = N_3 = 0.3333 + (1/15) * ecc**2
    else if (a[1] > a[2]):
        ecc = (a[0]**2 / a[2]**2 - 1)**0.5
        if (a[1] > a[2] * 2):
            N_3 = (1 - ecc**2)/(ecc**3) * (ecc - np.arctan(ecc))
            N_1 = N_2 = 0.5*(1 - N_3)
        else:
            N_3 = 0.3333 + (2/15) * ecc**2
            N_1 = N_2 = 0.3333 - (1/15) * ecc**2
    else: raise ValueError("spheroids of non-uniaxial anisotropy do not have a closed form function.")
    pe_mix_1 = pe_ice + bvf*pe_ice*((pe_brine - pe_ice) / (pe_ice + (1-bvf)*N_1*(pe_brine - pe_ice)))
    pe_mix_2 = pe_ice + bvf*pe_ice*((pe_brine - pe_ice) / (pe_ice + (1-bvf)*N_2*(pe_brine - pe_ice)))
    pe_mix_3 = pe_ice + bvf*pe_ice*((pe_brine - pe_ice) / (pe_ice + (1-bvf)*N_3*(pe_brine - pe_ice)))
    return np.array((pe_mix_1, pe_mix_2, pe_mix_3))

# Maxwell-Garnett, Bruggeman formula reference - https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-33-7-1244&id=344272#e18

def MaxwellGarnett_pemix(bvf, T, S, wavelength, f_pebrine='GW2020'):
    '''bvf: brine volume fraction\nT: temperature, celsius\nS: salinity, psu\nwavelength: metres'''
    k0 = 2*np.pi/wavelength
    
    #permittivities:
    pe_ice = 3.15 + 0.01j #ice permittivity largely insensitive to frequency @ L-band and above

    if (f_pebrine == 'GW2020'):
        pe_brine = GW2020_pebrine(T,S, wavelength)
    else: raise Exception('brine permittivity formula could not be found. Available formulas are : GW2020')

    #spherical inclusions mixing
    a = pe_brine + 2*pe_ice + 2*bvf*(pe_brine - pe_ice)
    b = pe_brine + 2*pe_ice - bvf*(pe_brine - pe_ice)
    pe_mix = pe_ice * a / b

    return pe_mix

    
def GW2020_pebrine(T, S, wavelength):
    '''T: temperature, Celsius \nS: salinity, psu'''
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
    #!IMPLEMENTED ------------------------------------------------------------------------------------------------------------------------- 
    # Seawater Debye Model Function at L-band and Its Impact on Salinity Retrieval from Aquarius Satellite Data
    # - https://ntrs.nasa.gov/api/citations/20210016626/downloads/TGRS-2020-01490.R1_Proof_seawater.pdf
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
    swmdw_ionic_R = 1 - S*(3.97185E-03 - 2.49205E-05*T - 4.27558E-05*S +  3.92825E-07*ST + 4.15350E-07*S**2)
    sw_ionic_R = 1+T*(3.76017e-2+6.32830e-5*T + 4.83420e-7*T**2 - 3.97484e-4*S + 6.26522e-6*S**2)
    sw_conductivityS0 = 9.50470e-2 * S - 4.30858e-4 * S**2 + 2.16182e-6 * S**3
    sw_conductivity = sw_conductivityS0 * sw_ionic_R
    sw_relaxation_time = 1.75030e-11 - 6.12993e-13*T + 1.24504e-14*T**2 - 1.14927E-16*T**3
    pe_brine = 4.9 + ((e_smdw*swmdw_ionic_R - 4.9) / (1+1j*wavelength*sw_relaxation_time)) - (1j*sw_conductivity / (wavelength*pe_0))
    return pe_brine


# An analysis of the tensor dielectric constant of sea ice at microwave frequencies
#   (Office of Naval Research Department of the Navy, Report 7975, October 1985)
# https://apps.dtic.mil/sti/tr/pdf/ADA168015.pdf
def Navy1985_pemix(bvf, T, S, wavelength):
    print("...later.")
    pass

def Stogryn1985_pebrine(T, S, wavelength):
    print("sorry, no luck here!")
    pass
    # The dielectric properties of brine in sea ice at microwave frequencies
    # Stogryn & Desargant 1985 (https://doi.org/10.1109/TAP.1985.1143610)
    #
    # "Unfortunately, all previously available descriptions of the dielectric constant of brine at the salinities and temperatures found in naturally
    #  occurring sea ice have been based on extra- polations of measurements made on NaCl solutions at temperatures above 0°C." <- too real, argh
    #
    # ... wait a second. There's no frequency samples at L-band.
    #      ...
    #      ..nooooooooooo whyyyyyyyyyyyyyyy