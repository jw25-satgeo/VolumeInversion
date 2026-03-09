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
"""Calculator module."""
import cmath
import math
import os
import random
import string
import sys

import numpy as np
import scipy.optimize as sc
from scipy.stats import lognorm


def initialise_unit_tensor():
    """Initialise a 3x3 tensor to a unit tensor.

    Parameters
    ----------
    None

    Returns
    -------
    array
        The returned tensor is a 3x3 array.

    """
    x = np.zeros((3, 3), dtype=float)
    x[0, 0] = 1.0
    x[1, 1] = 1.0
    x[2, 2] = 1.0
    return x

def initialise_complex_diagonal_tensor(reals):
    """Initialise a complex 3x3 tensor with the given diagonal components.

    Parameters
    ----------
    reals : list
        A list of 3 real numbers for the diagonals.

    Returns
    -------
    array
        The returned tensor is a complex 3x3 array.

    """
    x = np.zeros((3, 3), dtype=complex)
    x[0, 0] = reals[0]
    x[1, 1] = reals[1]
    x[2, 2] = reals[2]
    return x

def initialise_diagonal_tensor(reals):
    """Initialise a real 3x3 tensor with the given diagonal components.

    Parameters
    ----------
    reals : list
        A list of 3 real numbers for the diagonals.

    Returns
    -------
    array
        The returned tensor is a real 3x3 array.

    """
    x = np.zeros((3, 3), dtype=float)
    x[0, 0] = reals[0]
    x[1, 1] = reals[1]
    x[2, 2] = reals[2]
    return x

def calculate_distance(a,b):
    """Calculate the distance between a and b.

    Parameters
    ----------
    a : list of reals
        Coordinates of a
    b : list of reals
        Coordinates of b

    Returns
    -------
    float
        The distance between a and b

    """
    a = np.array(a)
    b = np.array(b)
    c = a - b
    return np.linalg.norm(c)


def calculate_angle(a,b,c):
    """Calculate the angle between a, b and c in degrees.

    The bond is a-b-c, b is the central atom

    Parameters
    ----------
    a : list of reals
        Coordinates of a
    b : list of reals
        Coordinates of b
    c : list of reals
        Coordinates of c

    Returns
    -------
    float
        The angle between a-b-c in degrees

    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    ba = ba / np.linalg.norm(ba)
    bc = bc / np.linalg.norm(bc)
    theta = np.arccos(np.dot(ba,bc))
    return np.rad2deg(theta)

def calculate_torsion(a,b,c,d):
    """Calculate the torsion angle between a, b, c and d in degrees.

    Parameters
    ----------
    a : list of reals
        Coordinates of a
    b : list of reals
        Coordinates of b
    c : list of reals
        Coordinates of c
    d : list of reals
        Coordinates of d

    Returns
    -------
    float
        The torsion angle between a-b-c-d in degrees

    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    # There are 3 bonds
    a_ba = -1.0*(b - a)
    b_cb = c - b
    c_dc = d - c
    # Calculate the perpendicular to abc and bcd
    axb = np.cross( a_ba, b_cb)
    bxc = np.cross( c_dc, b_cb)
    abc = np.cross(axb,bxc)
    y = np.dot(abc,b_cb)/np.linalg.norm(b_cb)
    x = np.dot(axb,bxc)
    theta = np.arctan2(y,x)
    return np.rad2deg(theta)


def initialise_sphere_depolarisation_matrix():
    """Initialise a 3x3 tensor with the sphere depolarisation matrix, returns a tensor.

    Parmeters
    ---------
    None

    Returns
    -------
    ndarray
        A 3x3 tensor representing the sphere depolarisation matrix.

    """
    athird = 1.0 / 3.0
    tensor = initialise_diagonal_tensor([athird, athird, athird])
    return tensor / np.trace(tensor)

def initialise_plate_depolarisation_matrix(normal):
    """Initialise a 3x3 tensor with the plate depolarisation matrix, returns a tensor.

    Parameters
    ----------
    normal : np.array
        A 3 vector giving the normal direction of the plate

    Returns
    -------
    ndarray
        A 3x3 tensor initialized with the plate depolarisation matrix.

    """
    normal = normal / np.linalg.norm(normal)
    tensor = np.outer(normal, normal)
    return tensor / np.trace(tensor)

def initialise_needle_depolarisation_matrix(unique):
    """Initialise a 3x3 tensor with the needle depolarisation matrix.

    Parameters
    ----------
    unique: np.array
        The unique direction of the needle

    Returns
    -------
    ndarray
        A 3x3 tensor initialized with the plate depolarisation matrix.

    """
    # unique is the unique direction of the needle
    # depolarisation matrix is therefore half the sum of the
    # two outer products of the directions perpendicular to it
    unique = unique / np.linalg.norm(unique)
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    xdotn = np.dot(x, unique)
    ydotn = np.dot(y, unique)
    zdotn = np.dot(z, unique)
    absxdotn = np.abs(xdotn)
    absydotn = np.abs(ydotn)
    abszdotn = np.abs(zdotn)
    # choose the direction with the smallest projection along the needle
    # then project out any needle direction
    if absxdotn <= absydotn and absxdotn <= abszdotn:
        dir1 = x - (xdotn * unique)
    elif absydotn <= absxdotn and absydotn <= abszdotn:
        dir1 = y - (ydotn * unique)
    else:
        dir1 = z - (zdotn * unique)
    dir1 = dir1/np.linalg.norm(dir1)
    # now find the orthogonal direction
    dir2 = np.cross(unique, dir1)
    dir2 = dir2/np.linalg.norm(dir2)
    # compute the complex tensors from the outer product of each direction
    tensor = np.outer(dir1, dir1) + np.outer(dir2, dir2)
    return tensor / np.trace(tensor)

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
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    xdotn = np.dot(x, unique)
    ydotn = np.dot(y, unique)
    zdotn = np.dot(z, unique)
    absxdotn = np.abs(xdotn)
    absydotn = np.abs(ydotn)
    abszdotn = np.abs(zdotn)
    # choose the direction with the smallest projection along the ellipsoid
    # then project out any ellipsoid direction
    if absxdotn <= absydotn and absxdotn <= abszdotn:
        dir1 = x - (xdotn * unique)
    elif absydotn <= absxdotn and absydotn <= abszdotn:
        dir1 = y - (ydotn * unique)
    else:
        dir1 = z - (zdotn * unique)
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
    else:
        nz = 1.0/3.0
    nxy = (1 - nz) * 0.5
    #
    # compute the tensors from the outer product of each direction
    tensor = nz*np.outer(unique, unique) + nxy*np.outer(dir1, dir1) + nxy*np.outer(dir2, dir2)
    return tensor / np.trace(tensor)

def fibonacci_sphere(samples=1,randomize=True):
    """Generate points on a sphere using the Fibonacci lattice method.

    This method generates points on the surface of a sphere in a way that they are evenly distributed. The distribution follows
    the Fibonacci Sphere algorithm, which is useful for creating well-distributed points across the surface of a sphere.
    Optionally, the points can be randomized slightly to avoid patterns.

    Parameters
    ----------
    samples : int, optional
        The number of points to generate on the sphere. Default is 1.
    randomize : bool, optional
        If `True`, the points generation will include a random factor to slightly change the distribution of points.
        This can be useful to avoid symmetric patterns. Default is `True`.

    Returns
    -------
    list
        A list of points where each point is represented as a list `[x, y, z]`, indicating its position on the sphere.

    Notes
    -----
    - The algorithm works by placing points at equal distances along a spiral around the sphere.
    - The `randomize` option applies a random offset to the spiral, which can help in reducing visual patterns in the distribution.

    Examples
    --------
    Generate 100 points on a sphere without randomization:

    >>> points = fibonacci_sphere(100, randomize=False)
    >>> len(points)
    100

    Generate 5 points on a sphere with default settings (including randomization):

    >>> points = fibonacci_sphere(5)
    >>> len(points)
    5

    """    
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = np.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return points



def normal_modes(masses, mass_weighted_normal_modes):
    """Transform from mass weighted coordinates to xyz.

    Note this returns an array object. The returned normal modes have NOT been renormalized.
    The input masses are in atomic units. The output normal modes are in atomic units.

    Parameters
    ----------
    masses : array of floats
        The masses in atomic units
    mass_weighted_normal_modes : array of floats
        The mass weighted normal modes

    Returns
    -------
    np.array
        An array of xyz coordinates derived from mass-weighted coordinates.

    Notes
    -----
    The transformation to xyz coordinates does not involve renormalization of the normal modes.
    Both the input masses and the output normal modes are in atomic units.

    """
    list_m = []
    normal_modes = np.zeros_like(mass_weighted_normal_modes)
    nions = np.size(masses)
    for a in range(nions):
        x = 1.0 / math.sqrt(masses[a])
        atom = [x, x, x]
        list_m.append(atom)
    # end of loop of ions
    array_m = np.array(list_m)
    for index, mode in enumerate(mass_weighted_normal_modes):
        normal_modes[index] = mode * array_m
    return normal_modes

def rodridgues_rotations(efield):
    """Take the field directions in efield and use each direction to calculate a random rotation about that axis.

    Use the field (which is a random unit vector in xyz space) to generate an orthogonal rotation matrix
    using the Rodrigues rotation formula A = I3.cos(theta) + (1-cos(theta)) e . eT + ex sin(theta),
    where I3 is a unit matrix, e is the direction, and ex is the cross product matrix:

    .. math::

        \\begin{align}
        &\\text{ex} = \\begin{pmatrix} 0 & -e3 & e2 \\\\ e3 & 0 & -e1 \\\\ -e2 & e1 & 0 \\end{pmatrix}
        \\end{align}

    Parameters
    ----------
    efield : array_like
        The field directions, where each direction is used to calculate a random rotation about that axis. Assumes the field is real.

    Returns
    -------
    list of ndarray
        A list of rotation matrices for each direction in efield.

    Notes
    -----
    - The input field is assumed to be real.
    - Output is a list of 3x3 rotation matrices.

    """
    rotations = []
    for field in efield:
        # Calculate a random angle between 0 and 180
        theta = np.pi*np.random.rand()
        cos   = np.cos(theta)
        sin   = np.sin(theta)
        rotation = np.zeros((3, 3))
        e1 = field[0]
        e2 = field[1]
        e3 = field[2]
        rotation[0, 0] = cos + (1-cos)*e1*e1
        rotation[0, 1] = (1-cos)*e2*e1 + sin*e3
        rotation[0, 2] = (1-cos)*e3*e1 - sin*e2
        rotation[1, 0] = (1-cos)*e1*e2 - sin*e3
        rotation[1, 1] = cos + (1-cos)*e2*e2
        rotation[1, 2] = (1-cos)*e3*e2 + sin*e1
        rotation[2, 0] = (1-cos)*e1*e3 + sin*e2
        rotation[2, 1] = (1-cos)*e2*e3 - sin*e1
        rotation[2, 2] = cos + (1-cos)*e3*e3
        rotations.append(rotation)
    return rotations

def absorption_from_mode_intensities(f, modes, frequencies, sigmas, intensities):
    """Calculate the absorption from the frequencies and intensities using a Lorentzian.

    Parameters
    ----------
    f : float
        The frequency of the absorption in cm-1.
    modes : list of ints
        A list of the modes 
    frequencies : list of reals
        Mode frequencies (cm-1)
    sigmas : list of reals
        Mode widths in cm-1
    intensities : list of reals
        Mode intensities (D2/A2/amu).

    Returns
    -------
    float
        The molar absorption coefficient at f, in L/mol/cm.

    Notes
    -----
    The number 4225.6 converts the units of D2/A2/amu to L mole-1 cm-1 cm-1.

    """
    absorption = 0.0
    for mode in modes:
        v = np.real(frequencies[mode])
        sigma = sigmas[mode]
        icastep = intensities[mode]
        absorption = absorption + 2.0 * 4225.6 * icastep / np.pi * (sigma / (4.0 * (f - v)*(f - v) + sigma*sigma))
    return absorption

def calculate_size_factor (x):
    """Calculate a size effect using Equations 10.38 and 10.39 in Sihvola.

    If x is small the result is  close to 1

    Parameters
    ----------
    x : float
       The size of the particle
 
    Returns
    -------
    float
        Size effect

    """
    if x < 1.0E-12:
        result = 1.0
    else:
        ix = complex(0,x)
        g1 = 2.0 / 3.0 * ( ( 1.0 + ix ) * np.exp(-ix) - 1.0 )
        g2 = ( 1 + ix - (7.0/15.0) * x*x - complex(0,2.0*x*x*x/15.0) ) * np.exp(-ix) -1.0
        result = 1 - g1 - g2
    return result

def averaged_permittivity(dielectric_medium, crystal_permittivity, shape, L, vf, size):
    """Calculate the effective constant permittivity using the averaged permittivity method.

    Parameters
    ----------
    dielectric_medium : array_like (3x3)
        The dielectric constant tensor of the medium.
    crystal_permittivity : array_like (3x3)
        The total frequency dielectric constant tensor at the current frequency.
    shape : str
        The name of the current shape.
    L : array_like (3x3)
        The shape's depolarisation matrix.
    vf : float
        Volume fraction
    size : float
        The dimensionless size parameter for the frequency under consideration (not used).

    Returns
    -------
    array_like
        The effective dielectric constant.

    """
    effd = vf * crystal_permittivity + (1.0-vf) * dielectric_medium
    trace = np.trace(effd) / 3.0
    return np.array([[trace, 0, 0], [0, trace, 0], [0, 0, trace]])

def balan(dielectric_medium, crystal_permittivity, shape, L, vf, size):
    """Calculate the effective constant permittivity using the method of Balan.

    Parameters
    ----------
    dielectric_medium : array_like (3x3)
        The dielectric constant tensor of the medium.
    crystal_permittivity : array_like (3x3)
        The total frequency dielectric constant tensor at the current frequency.
    shape : str
        The name of the current shape.
    L : array_like (3x3)
        The shape's depolarisation matrix.
    vf : float
        Volume fraction
    size : float
        The dimensionless size parameter for the frequency under consideration (not used).

    Returns
    -------
    array_like
        The effective dielectric constant.

    """
    unit = initialise_unit_tensor()
    dielecvm1 = (crystal_permittivity - unit)
    deformation  = np.dot(dielectric_medium, np.linalg.inv(dielectric_medium + np.dot(L, (crystal_permittivity - dielectric_medium))))
    effd          = unit + np.dot(deformation, dielecvm1)
    trace = vf * np.trace(effd) / 3.0
    return np.array([[trace, 0, 0], [0, trace, 0], [0, 0, trace]])

def maxwell(dielectric_medium, crystal_permittivity, shape, L, vf, size):
    """Calculate the effective constant permittivity using the Maxwell Garnett method.

    Parameters
    ----------
    dielectric_medium : tensor (3x3)
        The dielectric constant tensor of the medium.
    crystal_permittivity : tensor (3x3)
        The total frequency dielectric constant tensor at the current frequency.
    shape : str
        The name of the current shape.
    L : matrix (3x3)
        The shape's depolarisation matrix.
    size : float
        The dimensionless size parameter for the frequency under consideration.
    vf : float
        The volume fraction of filler.

    Returns
    -------
    tensor (3x3)
        The effective dielectric constant.

    """
    unit = initialise_unit_tensor()
    emedium = np.trace(dielectric_medium) / 3.0
    # If appropriate calculate a size effect using Equations 10.38 and 10.39 in Sihvola
    size_factor = calculate_size_factor(size)
    # Equation 5.78 in Sihvola
    nalpha = emedium*vf*np.dot((crystal_permittivity - dielectric_medium), np.linalg.inv(dielectric_medium + size_factor * np.dot(L, (crystal_permittivity-dielectric_medium))))
    nalphal = np.dot((nalpha/emedium), L)
    # average the polarisability over orientation
    nalpha = np.trace(nalpha) / 3.0 * unit
    # average the polarisability*L over orientation
    nalphal = np.trace(nalphal) / 3.0 * unit
    polarisation = np.dot(np.linalg.inv(unit - nalphal), nalpha)
    effd         = dielectric_medium + polarisation
    trace = np.trace(effd) / 3.0
    return np.array([[trace, 0, 0], [0, trace, 0], [0, 0, trace]])

def maxwell_sihvola(dielectric_medium, crystal_permittivity, shape, L, vf, size):
    """Calculate the effective constant permittivity using the Maxwell Garnett method.

    Parameters
    ----------
    dielectric_medium : tensor (3x3)
        The dielectric constant tensor of the medium.
    crystal_permittivity : tensor (3x3)
        The total frequency dielectric constant tensor at the current frequency.
    shape : str
        The name of the current shape.
    L : matrix (3x3)
        The shape's depolarisation matrix.
    size : float
        The dimensionless size parameter for the frequency under consideration.
    vf : float
        The volume fraction of filler.

    Returns
    -------
    tensor
        The effective dielectric constant.

    """
    unit = initialise_unit_tensor()
    # Equation 6.29 on page 123 of Sihvola
    # Equation 6.40 gives the averaging over the orientation function
    # See also equation 5.80 on page 102 and equation 4.31 on page 70
    Me = dielectric_medium
    # assume that the medium is isotropic calculate the inverse of the dielectric
    Mem1 = 3.0 / np.trace(Me)
    Mi = crystal_permittivity
    # If appropriate calculate a size effect using Equations 10.38 and 10.39 in Sihvola
    size_factor = calculate_size_factor(size)
    # calculate the polarisability matrix x the number density of inclusions
    nA = vf*np.dot((Mi-Me), np.linalg.inv(unit + (size_factor * Mem1 * np.dot(L, (Mi - Me)))))
    nAL = np.dot((nA), L)
    # average the polarisability over orientation
    nA = np.trace(nA) / 3.0 * unit
    # average the polarisability*L over orientation
    nAL = np.trace(nAL) / 3.0 * Mem1 * unit
    # Calculate the average polarisation factor which scales the average field
    # based on equation 5.80
    # <P> = pol . <E>
    pol = np.dot(np.linalg.inv(unit - nAL), nA)
    # Meff . <E> = Me . <E> + <P>
    # Meff . <E> = Me. <E> + pol . <E>
    # Meff = Me + pol
    effd         = dielectric_medium + pol
    # Average over orientation
    trace = np.trace(effd) / 3.0
    return np.array([[trace, 0, 0], [0, trace, 0], [0, 0, trace]])

def coherent(dielectric_medium, crystal_permittivity, shape, L, vf, size, dielectric_apparent):
    """Calculate the effective constant permittivity using the Coherent method.

    Parameters
    ----------
    dielectric_medium : tensor (3x3)
        The dielectric constant tensor of the medium.
    crystal_permittivity : tensor (3x3)
        The total frequency dielectric constant tensor at the current frequency.
    shape : str
        The name of the current shape.
    L : matrix (3x3)
        The shape's depolarisation matrix.
    size : float
        The dimensionless size parameter for the frequency under consideration.
    vf : float
        The volume fraction of filler.
    dielectric_apparent : 3x3 array of floats
        The current estimate of the dielectric

    Returns
    -------
    tensor
        The effective dielectric constant.

    """
    for _i in range(10):
        dielectric_apparent = 0.1 * dielectric_apparent + 0.9 * coherent2(dielectric_medium, dielectric_apparent, crystal_permittivity, shape, L, vf, size)
    return dielectric_apparent

def coherent2(dielectric_medium, dielectric_apparent, crystal_permittivity, shape, L, vf, size):
    """Calculate the effective constant permittivity using the Coherent2 method.

    Parameters
    ----------
    dielectric_medium : tensor (3x3)
        The dielectric constant tensor of the medium.
    dielectric_apparent : 3x3 array of floats
        The current estimate of the dielectric
    crystal_permittivity : tensor (3x3)
        The total frequency dielectric constant tensor at the current frequency.
    shape : str
        The name of the current shape.
    L : matrix (3x3)
        The shape's depolarisation matrix.
    size : float
        The dimensionless size parameter for the frequency under consideration.
    vf : float
        The volume fraction of filler.

    Returns
    -------
    tensor (3x3)
        The effective dielectric constant.

    """
    unit = initialise_unit_tensor()
    emedium = np.trace(dielectric_medium) / 3.0
    eapparent = np.trace(dielectric_apparent) / 3.0
    # If appropriate calculate a size effect using Equations 10.38 and 10.39 in Sihvola
    size_factor = calculate_size_factor(size)
    # Equation 5.78 in Sihvola
    nalpha = emedium*vf*np.dot((crystal_permittivity - dielectric_medium), np.linalg.inv(dielectric_medium + size_factor * np.dot(L, (crystal_permittivity-dielectric_medium))))
    nalphal = np.dot((nalpha/eapparent), L)
    # average the polarisability over orientation
    nalpha = np.trace(nalpha) / 3.0 * unit
    # average the polarisability*L over orientation
    nalphal = np.trace(nalphal) / 3.0 * unit
    polarisation = np.dot(np.linalg.inv(unit - nalphal), nalpha)
    effd         = dielectric_medium + polarisation
    trace = np.trace(effd) / 3.0
    return np.array([[trace, 0, 0], [0, trace, 0], [0, 0, trace]])

def bruggeman_minimise( eps1, eps2, shape, L, f2, size, epsbr):
    """Calculate the effective constant permittivity using the method of Bruggeman (minimisation).

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
    # we need to fool the optimiser into thinking that it has two real variables
    # in fact the second is imaginary and reconstructed in the _brug_minimise routine
    trace = np.trace(epsbr) / 3.0
    variables = np.array([np.real(trace), np.log(1.0 + np.abs(np.imag(trace)))])
    options = {"xtol": 1.0e-4,
               "ftol": 1.0E-4}
    sol = sc.minimize(_brug_minimise_tensor, variables, method="Powell", args=(eps1, eps2, shape, L, f1, size), options=options)
    if not sol.success:
        print("A Bruggeman solution was not found at this frequency")
    variables = sol.x
    # transform the imaginary variable back
    trace = complex(variables[0], np.exp(variables[1])-1.0)
    return np.array([[trace, 0, 0], [0, trace, 0], [0, 0, trace]])

def bruggeman_iter( eps1, eps2, shape, L, f2, size, epsbr, enforce_isotropy=False):
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
        epsbr, error = _brug_iter_error(epsbr, eps1, eps2, shape, L, f1, size, enforce_isotropy)
        if abs(error) < 1.0E-8:
            converged = True
        if niters > 3000:
            print("Bruggeman iterations failed, error=", error)
            converged = True
    return epsbr

def average_tensor(t):
    """Return the averaged tensor.

    Parameters
    ----------
    t : tensor (3x3)
        The tensor

    Returns
    -------
    output : tensor
        The averaged tensor.

    """
    a = np.trace(t) / 3.0
    return np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])

def _brug_minimise_scalar(variables, eps1, eps2, shape, L, f1, size):
    """Bruggeman method using scalar quantities (suitable for powell minimisation method).

    Parameters
    ----------
    variables : list of 2 floats
        The real and imaginary components of the scalar permittivity
    eps1      : tensor (3x3)
        Permittivity of phase 1
    eps2      : tensor (3x3)
        Permittivity of phase 2
    shape      : string
        The particle shape
    L         : 3x3 tensor
        Depolarisation tensor
    f1        : float
        The volume fraction of phase 1
    size      : float
        The size of the particle

    Returns
    -------
    float : The error associated with the current values of the permittivities

    """
    # unpack the complex number from the variables
    # two things going on here.
    # 1. the two variables refer to the real and imaginary components
    # 2. we require the imaginary component to be positive
    trace = complex(variables[0], np.exp(variables[1])-1.0)
    epsbr = np.array([[trace, 0, 0], [0, trace, 0], [0, 0, trace]])
    f2 = 1.0 - f1
    # If appropriate calculate a size effect using Equations 10.38 and 10.39 in Sihvola
    size_factor = calculate_size_factor(size)
    b1 = np.dot(L, (eps1 - epsbr))
    b2 = size_factor * np.dot(L, (eps2 - epsbr))
    tb1 = np.trace(b1)/3.0
    tb2 = np.trace(b2)/3.0
    ta1 = 1.0/(1.0 + tb1)
    ta2 = 1.0/(1.0 + tb2)
    c1 = eps1-epsbr
    c2 = eps2-epsbr
    tc1 = np.trace(c1)/3.0
    tc2 = np.trace(c2)/3.0
    # alpha1 and 2 are the polarisabilities of 1 and 2 in the effective medium
    talpha1 = tc1 * ta1
    talpha2 = tc2 * ta2
    error = f1*talpha1 + f2*talpha2
    error = np.abs(error.conjugate() * error)
    # Nasty issue in the powell method, the convergence on tol is given
    # relative to the solution (0.0).  Only a small number is added.
    # So we shift the solution by 1.0, the tol is now relative to 1.0
    return 1.0+error

def _brug_minimise_tensor(variables, eps1, eps2, shape, L, f1, size):
    """Bruggeman method using tensor quantities.

    Parameters
    ----------
    variables : list of 2 floats
        The real and imaginary components of the scalar permittivity
    eps1      : tensor (3x3)
        Permittivity of phase 1
    eps2      : tensor (3x3)
        Permittivity of phase 2
    shape      : string
        The particle shape
    L         : 3x3 tensor
        Depolarisation tensor
    f1        : float
        The volume fraction of phase 1
    size      : float
        The size of the particle

    Returns
    -------
    float : The error associated with the current values of the permittivities

    """
    # unpack the complex number from the variables
    # two things going on here.
    # 1. the two variables refer to the real and imaginary components
    # 2. we require the imaginary component to be positive
    trace = complex(variables[0], np.exp(variables[1])-1.0)
    epsbr = np.array([[trace,  0, 0], [0, trace, 0], [0, 0, trace]])
    f2 = 1.0 - f1
    # If appropriate calculate a size effect using Equations 10.38 and 10.39 in Sihvola
    size_factor = calculate_size_factor(size)
    b1 = np.dot(L, (eps1 - epsbr))
    b2 = size_factor * np.dot(L, (eps2 - epsbr))
    b1 = average_tensor(b1)
    b2 = average_tensor(b2)
    a1 = np.linalg.inv(epsbr + b1)
    a2 = np.linalg.inv(epsbr + b2)
    c1 = eps1-epsbr
    c2 = eps2-epsbr
    # c1 = average_tensor(eps1-epsbr)
    # c2 = average_tensor(eps2-epsbr)
    alpha1 = np.dot(c1, a1)
    alpha2 = np.dot(c2, a2)
    alpha1 = average_tensor(alpha1)
    alpha2 = average_tensor(alpha2)
    error = f1*alpha1 + f2*alpha2
    fr  = 0.0
    fi  = 0.0
    for c in error:
        for f in c:
            fr += np.real(f)**2
            fi += np.imag(f)**2
    error = np.linalg.norm(error)
    # Nasty issue in the powell method, the convergence on tol is given
    # relative to the solution (0.0).  Only a small number is added.
    # So we shift the solution by 1.0, the tol is now relative to 1.0
    return 1.0+error

def _brug_iter_error(epsbr, eps1, eps2, shape, L, f1, size, enforce_isotropy=False):
    """Routine to calculate the error in the Bruggeman method.

    Parameters
    ----------
    epsbr : list of 2 floats
        The real and imaginary components of the scalar permittivity
    eps1      : tensor (3x3)
        Permittivity of phase 1
    eps2      : tensor (3x3)
        Permittivity of phase 2
    shape : string
        The shape descriptor
    L         : 3x3 tensor
        Depolarisation tensor
    f1        : float
        The volume fraction of phase 1
    size      : float
        The size of the particle

    Returns
    -------
    float : The error associated with the current values of the permittivities

    """
    f2 = 1.0 - f1
    # If appropriate calculate a size effect using Equations 10.38 and 10.39 in Sihvola
    size_factor = calculate_size_factor(size)
    leps1 = np.dot(L, (eps1 - epsbr))
    leps2 = size_factor * np.dot(L, (eps2 - epsbr))
    if (enforce_isotropy): leps1, leps2 = average_tensor(leps1), average_tensor(leps2)
    a1 = np.linalg.inv(epsbr + leps1)
    a2 = np.linalg.inv(epsbr + leps2)
    # alpha1 and 2 are the polarisabilities of 1 and 2 in the effective medium
    if (enforce_isotropy): eps1av, eps2av = average_tensor(eps1), average_tensor(eps2)
    else: eps1av, eps2av = eps1, eps2
    alpha1 = np.dot((eps1av-epsbr), a1)
    alpha2 = np.dot((eps2av-epsbr), a2)
    # the error or residual matrix should be zero for a bruggeman solution
    error = f1*alpha1 + f2*alpha2
    error = np.linalg.norm(error)
    m1 = f1*np.dot(eps1, a1)+f2*np.dot(eps2, a2)
    m2 = np.linalg.inv(f1*a1 + f2*a2)
    damp = 0.0
    epsbr = (1.0 - damp)*np.dot(m1, m2) + damp*epsbr
    if (enforce_isotropy):
        trace = np.trace(epsbr) / 3.0
        epsbr = np.array([[trace, 0, 0], [0, trace, 0], [0, 0, trace]])
    return epsbr, error

def calculate_permittivity(refractive_index, debug=False):
    """Calculate the permittivity from the refractive index.

    Parameters
    ----------
    refractive_index : complex
        The refractive index from which the permittivity is calculated.
    debug : boolean
        True for debugging information

    Returns
    -------
    complex
        The calculated permittivity.

    """
    return refractive_index*refractive_index

def calculate_refractive_index(dielectric, debug=False):
    """Calculate the refractive index from the dielectric constant.

    Calculate the trace of the dielectric and calculate both square roots.
    Then choose the root with the largest imaginary component. This obeys the Kramers-Konig requirements.

    Parameters
    ----------
    dielectric : complex
        The permittivity
    debug : boolean
        True for debugging information

    Returns
    -------
    complex
        The refractive index calculated from the dielectric constant.

    Notes
    -----
    The calculation of the refractive index from the dielectric constant involves
    the trace of the dielectric tensor and the selection of the square root with
    the largest imaginary component.

    """
    trace = np.trace(dielectric)/3.0
    return calculate_refractive_index_scalar(trace, debug)

def calculate_refractive_index_scalar(dielectric_scalar, debug=False):
    """Calculate the refractive index from the dielectric constant.

    Calculate the trace of the dielectric and calculate both square roots.
    Then choose the root with the largest imaginary component. This obeys the Konig-Kramer requirements.

    Parameters
    ----------
    dielectric_scalar : complex
        The permittivity
    debug : boolean
        True for debugging information

    Returns
    -------
    complex
        The refractive index calculated from the dielectric constant.

    Notes
    -----
    The calculation of the refractive index from the dielectric constant involves
    the trace of the dielectric tensor and the selection of the square root with
    the largest imaginary component to satisfy the Konig-Kramer conditions.

    """
    solution1 = np.sqrt(dielectric_scalar)
    r, phase = cmath.polar(solution1)
    solution2 = cmath.rect(-r, phase)
    real1 = np.real(solution1)
    real2 = np.real(solution2)
    imag1 = np.imag(solution1)
    imag2 = np.imag(solution2)
    if abs(imag1)+abs(imag2) > 1.0e-18:
        solution = solution1 if imag1 > imag2 else solution2
    elif real1 > real2:
       solution = solution1
    else:
       solution = solution2
    if np.abs(solution*solution-dielectric_scalar)/(1+np.abs(dielectric_scalar)) > 1.0E-8 or debug:
        print("There is an error in refractive index")
        print("Dielectric = ", dielectric_scalar)
        print("solution*solution = ", solution*solution, np.abs(solution*solution-dielectric_scalar))
        print("solution    = ", solution, solution*solution)
        print("solution1   = ", solution1, solution1*solution1)
        print("solution2   = ", solution2, solution2*solution2)
    return solution

def direction_from_shape(data, reader):
    """Determine the unique direction of the shape from the shape data.

    Parameters
    ----------
    data : list of strings
        Data may contain a miller indices which defines a surface, e.g., (1,1,-1), 
        or a direction as a miller direction vector, e.g., [1,0,-1].
    reader : a reader objecy
        The reader is used to get the unit-cell

    Returns
    -------
    out : vector (3)
        Description of the unique direction determined from the data.

    """
    surface = False
    # original = data
    i = data.find(",")
    commas = False
    if i >= 0:
        commas = True
    if data[0] == "{":
        surface = True
        data = data.replace("{", "")
        data = data.replace("}", "")
    elif data[0] == "(":
        surface = True
        data = data.replace("(", "")
        data = data.replace(")", "")
    elif data[0] == "[":
        surface = False
        data = data.replace("[", "")
        data = data.replace("]", "")
    else:
        print("Error encountered in interpretting the miller surface / vector", data)
        sys.exit(1)
    if commas:
        data = data.replace(",", " ")
        hkl = ([int(f) for f in data.split()])
    else:
        i = 0
        hkl = [0, 0, 0]
        for k in [0, 1, 2]:
            sign = 1
            if data[i] == "-":
                i += 1
                sign = -1
            hkl[k] = sign * int(data[i])
            i += 1
    # end of handling no commas
    if len(hkl) != 3:
        print("Error encountered in interpretting the miller surface / vector", data)
        sys.exit(1)
    cell = reader.get_unit_cell()
    direction = cell.convert_hkl_to_xyz(hkl) if surface else cell.convert_abc_to_xyz(hkl)
    return direction / np.linalg.norm(direction)

def solve_effective_medium_equations( 
        method                     ,
        vf                         ,
        size_mu                    ,
        size_distribution_sigma    ,
        matrixPermittivityFunction ,
        shape                      ,
        L                          ,
        concentration              ,
        atrPermittivity            ,
        atrTheta                   ,
        atrSPol                    ,
        bubble_vf                  ,
        bubble_radius              ,
        previous_solution_shared   ,
        atuple                     ,
        ):
    """Solve the effective medium equations.

    Parameters
    ----------
    method : str
        The method to be used, options include bruggeman, balan, maxwell, maxwell-garnet, averagedpermittivity, maxwell-sihvola, coherent, bruggeman-minimise
    vf : float
        The volume fraction of dielectric.
    size_mu : float
        The particle size in micron
    size_distribution_sigma : float
        The width of the size distribution.
    matrixPermittivityFunction : function
        Function returning the matrix permittivity at a frequency.
    shape : str
        The shape of the particles.
    L : array
        The depolarisation matrix.
    concentration : float
        The concentration of particles.
    atrPermittivity : float
        The permittivity of the ATR substrate.
    atrTheta : float
        The ATR angle of incidence.
    atrSPol : str
        The ATR polarisation.
    bubble_vf : float
        Volume fraction of bubbles.
    bubble_radius : float
        The radius of bubbles.
    previous_solution_shared : bool
        Use the previous solution to speed up iterations in the case of Bruggeman and coherent methods.
    atuple : tuple
        A tuple containing frequency in cm-1 (v_cm1) and a rank 3 tensor of the permittivity of the crystal at a given frequency (crystalPermittivity).

    Returns
    -------
    Tuple of results
        v_cm1
        method
        size_mu
        size_distribution_sigma
        shape
        data
        trace
        absorption_coefficient
        molar_absorption_coefficient
        spatr

    """
    # unpack the tuple that is passed by a call to the partial function
    (v_cm1,crystalPermittivity) = atuple
    # convert the size to a dimensionless number which is 2*pi*size/lambda
    lambda_mu = 1.0E4 / (v_cm1 + 1.0e-12)
    size_mu = max(size_mu, 1.0e-12)
    size = 2.0*np.pi*size_mu / lambda_mu
    data = ""
    # Calculate the permittivity of the matrix as an isotropic tensor at v_cm1
    dielectric_medium = matrixPermittivityFunction(v_cm1) * np.eye(3)
    # Calculate the crystal permittivity at this frequency
    crystal_permittivity= crystalPermittivity
    # Calculate the effect of bubbles in the matrix by embedding in dielectric medium
    refractive_index = calculate_refractive_index(dielectric_medium, debug=False)
    if refractive_index.imag < 0.0:
        refractive_index = refractive_index.conjugate()
    if bubble_vf > 0.0:
        if np.abs(refractive_index.imag) > 1.0e-12:
            print("Warning: only the real part of the support matrix permittivity will be used for Mie Scattering",file=sys.stderr)
        effdielec,refractive_index = calculate_bubble_refractive_index(v_cm1, refractive_index.real, bubble_vf, bubble_radius)
        dielectric_medium = effdielec
    # Choose which method to apply, the effective dielectric determined with bubbles will be used
    if method == "balan":
        effdielec = balan(dielectric_medium, crystal_permittivity, shape, L, vf, size)
    elif method in ( "ap", "averagedpermittivity", "averaged permittivity", "average permittivity" ):
            effdielec = averaged_permittivity(dielectric_medium, crystal_permittivity, shape, L, vf, size)
    elif method in ( "maxwell", "maxwell-garnett" ):
        effdielec = maxwell(dielectric_medium, crystal_permittivity, shape, L, vf, size)
    elif method == "maxwell_sihvola":
        effdielec = maxwell_sihvola(dielectric_medium, crystal_permittivity, shape, L, vf, size)
    elif method == "coherent":
        eff  = previous_solution_shared
        if np.abs(np.trace(eff)) < 1.0e-8:
            eff = maxwell(dielectric_medium, crystal_permittivity, shape, L, vf, size)
        effdielec = coherent(dielectric_medium, crystal_permittivity, shape, L, vf, size, eff)
        previous_solution_shared = effdielec
    elif method == "bruggeman_minimise":
        eff  = previous_solution_shared
        if np.abs(np.trace(eff)) < 1.0e-8:
            eff = maxwell(dielectric_medium, crystal_permittivity, shape, L, vf, size)
        effdielec = bruggeman_minimise(dielectric_medium, crystal_permittivity, shape, L, vf, size, eff)
        previous_solution_shared = effdielec
    elif method in ( "bruggeman", "bruggeman_iter"):
        eff  = previous_solution_shared
        if np.abs(np.trace(eff)) < 1.0e-8:
            eff = maxwell(dielectric_medium, crystal_permittivity, shape, L, vf, size)
        effdielec = bruggeman_iter(dielectric_medium, crystal_permittivity, shape, L, vf, size, eff)
        previous_solution_shared = effdielec
    else:
        print(f"Unkown dielectric method: {method}")
        sys.exit(1)
    # Average over all directions by taking the trace
    trace = (effdielec[0, 0] + effdielec[1, 1] + effdielec[2, 2]) / 3.0
    refractive_index = calculate_refractive_index(effdielec)
    #
    # absorption coefficient is calculated from the imaginary refractive index
    # see H.C. van de Hulst Light Scattering by Small Particles , page 267
    # This is different but related to Genzel and Martin Equation 16, Phys. Stat. Sol. 51(1972) 91-
    # I've add a factor of log10(e) because we need to assume a decadic Beer's law
    # units are cm-1
    absorption_coefficient = v_cm1 * 4*np.pi * np.imag(refractive_index) * math.log10(math.e)
    # units are cm-1 L moles-1
    molar_absorption_coefficient = absorption_coefficient / concentration / vf
    # calculate the ATR reflectance
    spatr = reflectance_atr(refractive_index,atrPermittivity,atrTheta,atrSPol)
    return v_cm1,method,size_mu,size_distribution_sigma,shape,data,trace,absorption_coefficient,molar_absorption_coefficient,spatr

def calculate_centre_of_mass(xyzs, masses):
   """Calculate centre of mass.

   Parameters
   ----------
   xyzs : list of xyz coordinates of the atoms
       The xyz coordinates
   masses : list of the atomic masses
       The list of atomic masses in amu

   Returns
   -------
   mass : float - The total mass
   cm   : vector (3) - the coordinates of the centre of mass

   """
   cm = np.zeros(3)
   mass = 0.0
   for m,xyz in zip(masses,xyzs):
       mass += m
       cm   += m*xyz
   cm /= mass
   return mass,cm

def orthogonalise_projection_operator(ps):
   """Orthogonalise the list of projection operators ps.

   Use Gramm Schmidt orthogonalisation to perform the operation

   Parameters
   ----------
   ps : list of operators
       The projection operators to orthogonalise.

   Returns
   -------
   out : list of operators
       The orthogonalised projection operators.

   Notes
   -----
   This function orthogonalises a set of projection operators.

   """
   # The projection operator has dimension [6,natoms*3]
   maxcyc = 10
   cycle = 0
   max_overlap = 1.0
   while cycle < maxcyc and max_overlap > 1.0E-8:
       cycle += 1
       max_overlap = 0.0
       for i,p in enumerate(ps):
           # Normalise the projection operator
           anorm = np.dot(p,p)
           shift = 0.0
           if anorm < 1.0e-12:
               shift = 1.0E-12
           p = p / np.sqrt(np.dot(p,p) + shift )
           ps[i] = p
           for j,q in enumerate(ps):
               if j > i:
                   # Gramm Schmidt orthogonoalisation
                   dotprod = np.dot(p,q)
                   ps[j] = q - dotprod*p
                   max_overlap = max(max_overlap, dotprod)
   if cycle >= maxcyc:
       print("WARNING Schmidt Orthogonalisation Failed", max_overlap)
       sys.exit()
   return ps

def construct_projection_operator(atoms, xyzs, masses, nats):
   """Construct the projection operator for the molecule defined by atoms, xyz, masses.

   Parameters
   ----------
   atoms : list strings
       The atom types
   xyzs : list of atom coordinates vector (3)
       The list of coordinates
   masses : list of atom masses
       The list of atomic masses in amu
   nats : int
       The number of atoms

   Returns
   -------
   type
       The projection operators for this molecule

   """
   mass,cm = calculate_centre_of_mass(xyzs,masses)
   # The projection operator has dimension number_of_constraints,natoms*3
   ps = np.zeros( (6,nats*3) )
   x = 0
   y = 1
   z = 2
   for i,mass,xyz in zip(atoms,masses,xyzs):
       ps[0,i*3+x] = math.sqrt(mass)
       ps[1,i*3+y] = math.sqrt(mass)
       ps[2,i*3+z] = math.sqrt(mass)
       # coordinates relative to the centre of mass
       relxyz = math.sqrt(mass) * (xyz - cm)
       # First rotations about x in the y/z plane
       # zz is really r*sin(theta), and sin(theta) = zz/r
       # yy is really r*cos(theta), and cos(theta) = yy/r
       ps[3,i*3+x] = 0.0
       ps[3,i*3+y] = -relxyz[z]
       ps[3,i*3+z] = +relxyz[y]
       # Next rotations about y in the x/z plane
       ps[4,i*3+x] = +relxyz[z]
       ps[4,i*3+y] = 0.0
       ps[4,i*3+z] = -relxyz[x]
       # Next rotations about z in the x/y plane
       ps[5,i*3+x] = -relxyz[y]
       ps[5,i*3+y] = +relxyz[x]
       ps[5,i*3+z] = 0.0
   return ps

def cleanup_symbol(s):
    """Return a true element from the symbol.

    Parameters
    ----------
    s : str
        The element symbol to be cleaned up

    Returns
    -------
    str
        The cleaned symbol

    """
    s = s.capitalize()
    s = s.replace("_","")
    for i in string.digits:
        s = s.replace(i,"")
    return s

def determineEulerAngles(R):
     """Determine the euler angles of a rotation matrix.

     Parameters
     ----------
     R : tensor (3,3)
        The rotation matrix

     Returns
     -------
     theta, phi, psi : float
        The Euler angles

     """
     R11=R[0,0]
     R12=R[0,1]
     R13=R[0,2]
     R21=R[1,0]
     R31=R[2,0]
     R32=R[2,1]
     R33=R[2,2]
     phi = 0.0
     if R31 > 0.9999999999:
         theta = -np.pi/2.0
         psi   = -phi + np.arctan2(-R12,-R13)
     elif R31 < -0.9999999999:
         theta = np.pi/2.0
         psi   = phi + np.arctan2(R12,R13)
     else:
         theta = theta1 = -np.arcsin(R31)
         psi = np.arctan2(R32/np.cos(theta1), R33/np.cos(theta1))
         phi = np.arctan2(R21/np.cos(theta1), R11/np.cos(theta1))
     return theta, phi, psi

def euler_rotation(vector, theta, phi, psi):
     """Apply a passive Euler rotation to a vector.

     Parameters
     ----------
     vector : vector (3)
         The vector to be rotated
     theta  : float
         The angle theta
     phi    : float
         The angle phi
     psi    : float
         The angle psi

     Returns
     -------
     vector (3)

     Notes
     -----
     A passive Euler rotation refers to the rotation of the coordinate system 
     while the vector remains fixed. This operation is often used in 
     physics and engineering to describe the orientation of an object 
     with respect to a reference coordinate system.

     """
     euler = np.zeros( (3,3) )
     euler[0, 0] =  np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi)
     euler[0, 1] = -np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi)
     euler[0, 2] =  np.sin(theta) * np.sin(phi)
     euler[1, 0] =  np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
     euler[1, 1] = -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi)
     euler[1, 2] = -np.sin(theta) * np.cos(phi)
     euler[2, 0] =  np.sin(theta) * np.sin(psi)
     euler[2, 1] =  np.sin(theta) * np.cos(psi)
     euler[2, 2] =  np.cos(theta)
     return np.matmul(euler, vector)

def get_pool(ncpus, threading, initializer=None, initargs=None, debugger=None ):
     """Return a pool of processors given the number of cpus and whether threading is requested.

     Parameters
     ----------
     ncpus : int 
        the number of processors
     threading : bool 
        true if threading is to be used
     initializer : function
        Function to be called before getting the pool
     initargs : function arguments
        Any other parameters
     debugger : a debugger object
        A debugger object

     Returns
     -------
     pool : the pool of processors

     """
     if debugger is not None:
         debugger.print("get_pool ncpus = ",ncpus)
         debugger.print("get_pool threading = ",threading)
         debugger.print("get_pool initializer = ",initializer)
     # see if threading has been requested
     if threading:
         from multiprocessing.dummy import Pool
         if debugger is not None:
             debugger.print("get_pool using the multiprocessing package and threading")
     else:
         from multiprocessing import Pool
         if debugger is not None:
             debugger.print("get_pool using the multiprocessing package")
     return Pool(ncpus, initializer=initializer, initargs=initargs)

def set_no_of_threads(nthreads):
    """Set default number of threads.

    Parameters
    ----------
    nthreads : int the number of threads to be used
        The number of threads to be applied

    The environment is modified to set the most common environment variables for the number of threads
    a BLAS implementation will use.
    BLAS implementations include: MKL, OPENBLAS, OMP, NUMEXPR, BLIS and VECLIB

    """
    os.environ["MKL_NUM_THREADS"]        = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"]   = str(nthreads)
    os.environ["OMP_NUM_THREADS"]        = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"]    = str(nthreads)
    os.environ["BLIS_NUM_THREADS"]       = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)

def compute_all_sg_permutations(rot,mat):
    """Similarity transformation by R x M x R^-1.

    This routine was taken from Phonopy.

    Parameters
    ----------
    rot : 3x3 floats
        The rotation matrix
    mat : 3x3 floats
        The matrix to be transformed

    """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))



def set_affinity_on_worker():
    """When a new worker process is created, the affinity is set to all CPUs."""
    #JK print("I'm the process %d, setting affinity to all CPUs." % os.getpid())
    #JK Commented out for the time being
    #JK os.system("taskset -p 0xff %d > /dev/null" % os.getpid())

def similarity_transform(rot,mat):
    """Similarity transformation by R x M x R^-1.

    This routine was taken from Phonopy.

    Parameters
    ----------
    rot : 3x3 floats
        The rotation matrix
    mat : 3x3 floats
        The matrix to be transformed

    """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))