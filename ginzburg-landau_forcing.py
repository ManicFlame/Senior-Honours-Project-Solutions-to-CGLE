"""
Program to find numerical solutions to the complex ginzburg-landau equation:

u_t = c_g * u_x + mu*u + aa*|u|**2 *u + lmbda *u_xx

Applies forcing at boundary x=l

Outputs: Colour-coded x-t plot indicating the value of Re(u)

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib import cm

"""
Function to construct a Chebyshev spectral differentiation matrix for a given number of points on the domain xi \elem [-1,1]
Input: Number of points, N
Outputs: First derivative Chebyshev matrix, (-)D
         Chebyshev grid points, xi
"""

def cheb_diff_matrix(N):
    if N == 0:
        return np.array([[0.]]), np.array([1.])
    
    xi = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c = c * ((-1) ** np.arange(N + 1))

    X = np.tile(xi, (N + 1, 1))
    dX = X - X.T

    D = (np.outer(c, 1/c)) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))

    # We have to return -D here since Trefethen's formula assumes the points on the grid are ordered the opposite way
    return -D, xi



def main():

    # Number of points in Chebyshev grid
    N = 149

    # CGL parameters: see top comment
    c_g = 1.0
    mu = 0.50
    lmbda = 1.0 + 0.45j

    #usual aa = -1.0 + 2.0j
    aa = -1.0 + 2.0j


    # Domain length x elem [0,l]
    l = 60

    # Forcing parameters
    #prev using amp = 0.1, omega = 1.0
    amp = 0.1
    omega = 2.0
    

    # Compute the first and second derivative Chebyshev matrix for xi elem [-1,1]
    D1_xi, xi = cheb_diff_matrix(N)
    D2_xi = D1_xi @ D1_xi


    # rescaled x for the domain [0,l] from xi elem [-1,1]
    x = 0.5 * l * (xi + 1) 

    # rescale D1 and D2 to fit the domain x elem [0,l]
    D1_x = 2.0 / l * D1_xi
    D2_x = ((2.0 / l)**2) * D2_xi


    # Cut off the first and last elements of the grid
    x = x[1:-1]


    # Apply initial conditions to v: the physical solution u = v + phi but phi at t=0 is zero, so it doesn't change
    v = 0.5 * np.sin((2 * np.pi * x)/ l)

    


    """
    Precomputuing various ETDRK4 values
    """

    h = 0.001; # time step
    M = 32; # no. of points for resolvent integral
    r = 15 * np.exp(1j * np.pi*(np.arange(1, M+1)- 0.5) / M)


    # The linear operator L which handles first and second derivative terms
    # first spatial derivative term: c_g * u_x with end points cut off
    # second spatial derivative term: lmbda * u_xx with end points cut off
    
    I = np.eye(N-1, dtype = np.complex128)
    
    L = c_g * D1_x[1:N, 1:N] + lmbda * D2_x[1:N, 1:N] + mu * I
    

    A = h * L
    E = expm(A)
    E2 = expm(A/2)

    I = np.eye(N-1, dtype = np.complex128)
   
    Q = np.zeros((N-1, N-1), dtype=np.complex128)
    f1 = np.zeros((N-1, N-1), dtype=np.complex128)
    f2 = np.zeros((N-1, N-1), dtype=np.complex128)
    f3 = np.zeros((N-1, N-1), dtype=np.complex128)

    for j in range(M):  
        z = r[j]                   
        zIA = np.linalg.inv(z * I - A) 
        
        # Accumulate matrices
        Q  += h * zIA * (np.exp(z/2) - 1)
        f1 += h * zIA * (-4 - z + np.exp(z)*(4 - 3*z + z**2)) / z**2
        f2 += h * zIA * (2 + z + np.exp(z)*(z - 2)) / z**2
        f3 += h * zIA * (-4 - 3*z - z**2 + np.exp(z)*(4 - z)) / z**2

    f1 = np.real(f1/M); f2 = np.real(f2/M); f3 = np.real(f3/M); Q = np.real(Q/M)

    """
    End of precomputing values
    """

    vlength = len(v)  # length of initial u vector
    print(vlength)

    # Initialize empty solution arrays for storage
    vv = np.empty((vlength, 0), dtype=np.complex128)   
    tt = np.empty((1, 0), dtype=np.float64)  


    # u is the physical solution for plotting: it has the same first part as vv since phi(t=0) == 0
    uu = np.empty((vlength+2,0), dtype=np.complex128)
    u = np.r_[0, np.copy(v), 0]




    # Add the first time step (initial condition)
    
    #  change to column vectors
    v = v.reshape(-1,1)  
    u = u.reshape(-1,1)      

    # vv now has shape (N,1): this adds a single column v and likewise with uu
    vv = np.hstack([vv, v])    
    uu = np.hstack([uu,u])

    # tt now has shape (1,1)
    tt = np.hstack([tt, np.array([[0]])])  

    tmax = 200
    nmax = np.round(tmax/h)
    nplt = np.floor((tmax/200)/h)
    print(f"nplt: {nplt}")
    
    
    for n in range(1, int(nmax+1)):
    
        t = n*h

        # 'lifting' function to make boundary conditions homogeneous
        phi = amp/l * x * np.sin(omega*t)
        phi = phi.reshape(-1,1) 

        phi_x = D1_x[1:N,1:N] @ phi
        phi_xx = 0

        phi_t = omega * (amp/l) * x * np.cos(omega*t)
        phi_t = phi_t.reshape(-1, 1)
        
        
        forcing =  c_g * phi_x + mu * phi + lmbda * phi_xx - phi_t
        
        Nv = aa * (np.absolute(v + phi)**2) * (v + phi)


        a =  E2 @ v + Q @ Nv + Q @ forcing
        #SHould include + mu* a etc, on Na Nb Nc
        Na =  aa * (np.absolute(a + phi)**2) * (a + phi)

        b =  E2 @ v + Q @ Na + Q @ forcing
        Nb =  aa * (np.absolute(b + phi)**2) * (b + phi)

        c = E2 @ a + Q @ (2*Nb - Nv) + Q @ forcing
        Nc = aa * (np.absolute(c + phi)**2) * (c + phi)

        # Update the interior values of v
        v = E @ v + f1 @ Nv + 2*f2 @ (Na + Nb) + f3 @ Nc + Q @ forcing
    

        # Plot every nplt steps
        
        if np.mod(n,nplt) == 0:
          
            print(f"{n} points reached")
           
            # concatenate onto vv as a new column  
            v = v.reshape(-1,1)        
            vv = np.hstack([vv, v])    

     
            # u = v + phi is the physical solution: we have its solution excluding the two endpoints x=0 and x=L, so these need added
            # need to start at x = l, end at x = 0
        

            u = np.vstack([amp*np.sin(omega*t), v + phi])
            u = np.vstack([u, 0])
    
            uu = np.hstack([uu,u])
            tt = np.hstack([tt, np.array([[t]])])  # row vector of times

        

    # Plotting Code

    uu_plot = np.real(uu)

    # x-axis: [l, x, 0] 
    X_axis = np.r_[l, x, 0] 

    #reverse the x array so it starts with 0
    print(f"X axis: {X_axis}")  

    # Time axis: tt will be flattened to a 1D array
    T_axis = tt.flatten() 
    print(f"T axis: {T_axis}")     

    # MATLAB uses uu' to mean transpose
    #would be uu_plot if line above used
    Z = uu_plot.T

    plt.pcolormesh(X_axis, T_axis, Z, shading="auto", cmap="viridis")

    plt.xlabel("x", fontsize=26)
    plt.ylabel("t", fontsize=26)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label("Re(A)", fontsize=26)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
   
    plt.show()  
    
main()


