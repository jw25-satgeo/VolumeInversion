import numpy as np



#################################################################

def carToSpherical(Nz=1, Ny=1, Nx=1, cz=0, cy=0, cx=0):
    x = np.arange(Nx)
    y = np.arange(Ny)
    z = np.arange(Nz)

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    # shift coordinates
    X_shifted, Y_shifted, Z_shifted = X -cx, Y - cy, Z - cz

    # Calculate spherical coordinates
    r = np.sqrt(Z_shifted**2 + Y_shifted**2 + X_shifted**2)
    phi = np.arctan2(Y_shifted, X_shifted)   
    theta = np.arctan2(np.sqrt(X_shifted**2 + Y_shifted**2), Z_shifted)

    return r, theta, phi
#################################################################



#################################################################

def sphere(r, a):
    return r <= a

#################################################################




###################################################################
##
##def circle(r, a, Ny=10, Nx=10, er1, er2, er):
##    er = 
##    inside = np.zeros((Ny, Nx), dtype=bool)
##    for i in range(Ny):
##        for j in range(Nx):
##            if r[i, j] <= a:
##                inside[i, j] = True
##    return inside
###################################################################






#################################################################

def square(a, Ny=10, Nx=10, cy=0, cx=0):
    inside = np.zeros((Ny, Nx), dtype=bool)
    for i in range(Ny):
        for j in range(Nx):
            Y = i - cy
            X = j - cx
            if (X <= a and X >= -a and Y <= a and Y >= -a):
                inside[i, j] = True
    return inside
#################################################################




#################################################################

def rectangle(a, b, theta, Ny=10, Nx=10, cy=0, cx=0):
    inside = np.zeros((Ny, Nx), dtype=bool)
    for i in range(Ny):
        for j in range(Nx):
            Y = np.cos(theta)*(j - cx) - np.sin(theta)*(i - cy)
            X = np.sin(theta)*(j - cx) + np.cos(theta)*(i - cy)
            if (X <= b and X >= -b and Y <= a and Y >= -a):
                inside[i, j] = True
    return inside
#################################################################




#################################################################

def hexagon(a, Ny=10, Nx=10, cy=0, cx=0):
    inside = np.zeros((Ny, Nx), dtype=bool)
    for i in range(Ny):
        for j in range(Nx):
            Y = i - cy
            X = j - cx
            if (Y <= X + 2*a and Y >= - X - 2*a and Y >= X - 2*a and Y <= -X + 2*a and X >= - a and X <= a):
                inside[i, j] = True
    return inside
#################################################################   




#################################################################

def corrugatedEllipse(r, phi, a, Ny=10, Nx=10, A=1, N=0, epsilon=0):
    inside = np.zeros((Ny, Nx), dtype=bool)
    for i in range(Ny):
        for j in range(Nx):
            if (r[i, j] <= a * (1 / ((np.cos(phi[i, j]))**2 + (A * np.sin(phi[i, j]))**2) + epsilon * np.cos(N * phi[i, j]))):
                inside[i, j] = True
    return inside
#################################################################

                


#################################################################

def JanusHalf(r, a, theta, Ny=10, Nx=10, cy=0, cx=0):
    first_half = np.zeros((Ny, Nx), dtype=bool)
    second_half = np.zeros((Ny, Nx), dtype=bool)
    d2r = np.pi/180
    for i in range(Ny):
        for j in range(Nx):
            Y =   np.cos(theta*d2r)*(j - cx) + np.sin(theta*d2r)*(i - cy)
            X = - np.sin(theta*d2r)*(j - cx) + np.cos(theta*d2r)*(i - cy)
            if (X <= 0 and r[i, j] <= a):
                first_half[i, j] = True
            elif (X > 0 and r[i, j] <= a):
                second_half[i, j] = True
    return first_half, second_half
#################################################################





#################################################################

def JanusSegment(r, er, a, er2, er3, Ny=10, Nx=10, cy=0, cx=0, shift=0):
    for i in range(Ny):
        for j in range(Nx):
            Y = i - cy
            X = j - cx
            if (r[i, j] <= a and X <= -(a-shift)):
                er[i, j]  = er2
            elif (r[i, j] <= a and X > -(a-shift)):
                er[i, j]  = er3
#################################################################




#################################################################

def JanusSector(r, er, a, er2, er3, Ny=10, Nx=10, cy=0, cx=0, theta=0):
    for i in range(Ny):
        for j in range(Nx):
            Y = i - cy
            X = j - cx
            if (r[i, j] <= a):
                if (theta <= 90):
                    if (X <= 0 and np.tan(theta * np.pi/180)*X <= Y <= -np.tan(theta * np.pi/180)*X):
                        er[i, j]  = er2
                    else :
                        er[i, j]  = er3
                else :
                    if (X > 0 and -np.tan((180-theta) * np.pi/180)*X <= Y <= np.tan((180-theta) * np.pi/180)*X):
                        er[i, j]  = er3
                    else :
                        er[i, j]  = er2
#################################################################














