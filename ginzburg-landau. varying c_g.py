"""
Program to find numerical solutions to the complex ginzburg-landau equation:

u_t = c_g * u_x + mu*u + aa*|u|**2 *u + lmbda *u_xx


Outputs: A cycle of plots of Re(A) vs. x for every 1 unit of time
         Saved plots of Re(A) vs. for every 50 units of time
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



    # Number of points in Chebyshev grid: this should be 1 less than however many points you want
    N = 149
    
    # CGL parameters: see top comment for reference
    c_g = 1.0
    mu = 0.2
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

    u = 0.5 * np.sin((2 * np.pi * x)/ l)
    
    

    """
    Precomputuing various ETDRK4 values
    """

    h = 1e-3; # time step
    M = 32; # no. of points for resolvent integral
    r = 15 * np.exp(1j * np.pi*(np.arange(1, M+1)- 0.5) / M)

    """
    The linear operator L which handles first and second derivative terms
    first spatial derivative term: c_g * u_x with end points cut off
    second spatial derivative term: lmbda * u_xx with end points cut off   
    """
 
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
    uu = np.empty((N, 0), dtype=np.complex128)   
    tt = np.empty((1, 0), dtype=np.float64)     


    # Add the first time step (initial condition) to the solutions array: to do this u must be a column vector
    u = u.reshape(-1,1)      

    # uu now has shape (N,1)
    uu = np.hstack([uu, u])    

    # tt now has shape (1,1)
    tt = np.hstack([tt, np.array([[0]])])  

    tmax = 400
    nmax = np.round(tmax/h)
    nplt = np.floor((tmax/400)/h)
    
    print(f"nplt: {nplt}")
    

    plt.ion()
    
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
    
        # Plot every nplt steps
        
        if np.mod(n,nplt) == 0:
          


            print(f"{n} points reached")
            
            
            plt.clf()
            plt.plot(x,u, linewidth=2.5)
            plt.xlabel("x", fontsize=22)
            plt.ylabel("Re(A)",fontsize=22)
            plt.title(f"t = {t}", fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.pause(0.1)
            

        if np.mod(t, 50) == 0:
            
            plt.clf()


            dpi = 100
            plt.figure(figsize=(1920/dpi, 1080/dpi))
            plt.plot(x,np.real(u), linewidth=2.5)

            plt.xlabel("x", fontsize=26)
            plt.ylabel("Re(A)",fontsize=26)
            #plt.title(f"t = {t}", fontsize=24)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)

            plt.tight_layout()

            plt.show()
            plt.savefig(f"plot{t}.png", dpi=dpi, bbox_inches='tight')

        
    plt.ioff()
    plt.show()

    
main()


