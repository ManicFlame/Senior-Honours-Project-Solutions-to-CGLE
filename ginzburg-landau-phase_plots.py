"""
Program to find numerical solutions to the complex ginzburg-landau equation:

u_t = c_g * u_x + mu*u + aa*|u|**2 *u + lmbda *u_xx


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting
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
    mu = 0.3
    lmbda = 1.0 + 0.45j
    aa = -1.0 + 2.0j


    # Domain length x elem [0,l]
    l = 60
    

    # Compute the first and second derivative Chebyshev matrix for xi elem [-1,1]
    D1_xi, xi = cheb_diff_matrix(N)
    D2_xi = D1_xi @ D1_xi


    # rescaled x for the domain [0,l] from xi elem [-1,1]
    x = 0.5 * l * (xi + 1) 

    # rescale D1 and D2 to fit the domain x elem [0,l]
    D1_x = 2.0 / l * D1_xi
    D2_x = ((2.0 / l)**2) * D2_xi


    # Cut off the first and last elements of the grid
    x = x[1:N]

    
    #apply initial conditions to u: here I've chosen a sine wave which correctly matches the boundary conditions

    u = 0.5 * np.sin((4 * np.pi * x)/ l)
    
    
    
    #w = 0.5 * np.sin((2 * np.pi * x)/ l) - x

    # should read x -> w+x if keeping w term. the endpoints here read 1 and then -1 in Trefethen - if these are BCs (as I've assumed) then set both to zero
    # starts at u(l) = 0, ends at u(0) = 0
    # u = np.r_[0, w+x, 0]


    """
    Precomputuing various ETDRK4 values
    """

    h = 0.001; # time step
    M = 32; # no. of points for resolvent integral
    r = 15 * np.exp(1j * np.pi*(np.arange(1, M+1)- 0.5) / M)


    # The linear operator L which handles first and second derivative terms
    # first spatial derivative term: c_g * u_x with end points cut off
    # second spatial derivative term: lmbda * u_xx with end points cut off
    L = c_g * D1_x[1:N, 1:N] + lmbda * D2_x[1:N, 1:N]
    

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

    
    N = len(u)  # length of initial u vector

    # Initialize empty arrays for storage
    uu = np.empty((N, 0), dtype=np.complex128)   # shape (N+2, 0) to account for adding back the endpoints
    tt = np.empty((1, 0), dtype=np.float64)      # shape (1, 0)

    phi = np.arctan(np.imag(u) / np.real(u))
    phiphi = np.empty((N,0), dtype=np.float64)



    # Add the first time step (initial condition)
    
     # column vector
    u = u.reshape(-1,1) 
    phi = phi.reshape(-1, 1)

    print(u.shape)
    print(phi.shape)

    # uu now has shape (N,1)
    uu = np.hstack([uu, u])    
    phiphi = np.hstack([phiphi, phi])

    # tt now has shape (1,1)
    tt = np.hstack([tt, np.array([[0]])])  

    tmax = 100
    nmax = np.round(tmax/h)
    nplt = np.floor((tmax/100)/h)
    print(f"nplt: {nplt}")
    
    
    for n in range(1, int(nmax+1)):
    
        t = n*h
        
        Nu = mu * u + aa * (np.absolute(u)**2) * u
        a =  E2 @ u + Q @ Nu
        Na = mu * a + aa * (np.absolute(a)**2) * a
        b =  E2 @ u + Q @ Na
        Nb = mu * b + aa * (np.absolute(b)**2) * b
        c = E2 @ a + Q @ (2*Nb - Nu)
        Nc = mu * c + aa * (np.absolute(c)**2) * c

        # Update the interior values of u
        u = E @ u + f1 @ Nu + 2*f2 @ (Na + Nb) + f3 @ Nc
    

        #plt.ion()
        # Plot every nplt steps
        if np.mod(n,nplt) == 0:
          
            
            print(n)

            phi = np.arctan(np.imag(u) / np.real(u))
        

           
            u = u.reshape(-1,1)      
            phi = phi.reshape(-1,1)

            uu = np.hstack([uu, u])        # concatenate as new column
            phiphi = np.hstack([phiphi, phi])
            tt = np.hstack([tt, np.array([[t]])])  # row vector of times


        #plt.ioff()
        #plt.show()
    
    # Plotting Code

    
    # Adds back in zero values of u at all the endpoints

    uu_plot = np.vstack([np.zeros((1, uu.shape[1])), np.real(uu), np.zeros((1, uu.shape[1]))])



    phiphiplot = phiphi.T

    print(phiphiplot)


    # currently both lists have x=l near at the 0th index end 

    #print(phiphiplot[0].shape)
    #print(x.shape)
    #print(phiphiplot[0])

    times = tt.flatten()
    print(times[0])
    
    plt.ion()

    
    for i, t in enumerate(times):

        plt.clf()

        plt.plot(x, phiphiplot[i])
        plt.xlabel("x")
        plt.ylabel("Phase")
        plt.title(f"t = {t}")

        plt.pause(0.5)

    plt.ioff()
    plt.show()    
    """
    #print(uu_plot)
    # uu_plot = np.vstack([np.zeros((1, uu.size)),  uu.reshape(1, -1).T, np.zeros((1, uu.size))])

    # x-axis: [l, x, 0] 
    X_axis = np.r_[l, x, 0]   

    # Time axis: tt will be flattened to a 1D array
    T_axis = tt.flatten()    

    # MATLAB uses uu' to mean transpose
    #would be uu_plot if line above used
    Z = uu_plot.T
    Z_plot = Z
             
 
    # Create meshgrid 
    X_mesh, T_mesh = np.meshgrid(X_axis, T_axis, indexing='ij')

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_mesh, T_mesh, Z_plot.T, cmap=cm.magma, edgecolor='none')

    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    ax.set_zlabel('abs(u)')
    ax.view_init(elev=60, azim=-45)

    # Stretch X and Y axes for better visualization if needed
    ax.set_xlim([np.min(X_axis), np.max(X_axis)])
    ax.set_ylim([np.min(T_axis), np.max(T_axis)])

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    """
    
main()


