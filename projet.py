"""
------------------------------------------------
 Ma324 - Optimisation différentiable 2 - Projet
------------------------------------------------
AYACH Tiffen
LOMBARDO Baptiste
MIAUX Romain
PREMARAJAH Piratheban
3PSA - 3PSC
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def f(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def grad(f):
    grad = np.zeros((2, 1))
    x = f[0]
    y = f[1]
    grad[0, 0] = 2 * (2*x * (x**2 + y - 11) + x + y**2 - 7)
    grad[1, 0] = 2 * (x**2 + 2*y * (x + y**2 - 7) + y - 11)
    return grad

def H(f):
    H = np.zeros((2, 2))
    x = f[0]
    y = f[1]
    H[0, 0] = 4 * (x**2 + y - 11) + 8*x**2 + 2
    H[0, 1] = 4*x + 4*y
    H[1, 0] = 4*x + 4*y
    H[1, 1] = 4 * (x + y**2 - 11) + 8*y**2 + 2
    return H

def Dj(f):
    Dj = np.zeros((2, 2))
    n = f.size
    for i in range(n):
        Dj[i, i] = H(f)[i, i]
    return Dj

     
# -------------------------------------------------------------------
#                       Méthodes des gradients
# -------------------------------------------------------------------


def gradientPasFixe(x0, rho, tol):
    
    NiterMax = 10**5
    n = 0
    xit = []
    
    d0 = -grad(x0)
    x1 = x0 + rho * d0
    n += 1
    
    while ((np.linalg.norm(grad(x0)) > tol) and (n < NiterMax)):
        xit.append(x1)
        x0 = x1
        d0 = -grad(x0)
        x1 = x0 + rho * d0
        n += 1
    
    sol = x1
    
    return sol, xit, n, np.linalg.cond(H(x0))


def rechercheDuPas(xi, di, rho0=10**-3, tolR=10**-8):
    
    NiterMax = 10**4
    n = 0
    rho0 = 10 * 10**-3
    
    phip = di.T @ grad(xi + rho0 * di)
    phipp = di.T @ H(xi + rho0 * di) @ di
    rhoN = rho0 - phip / phipp
    n += 1
    
    while ((np.linalg.norm(rhoN - rho0) > tolR) and (n < NiterMax)):
        rho0 = rhoN
        phip = di.T @ grad(xi + rho0 * di)
        phipp = di.T @ H(xi + rho0 * di) @ di
        rhoN = rho0 - phip / phipp
        n += 1
        
    return rhoN
    
    
def gradientPasOptimal(x0, tol):
    
    NiterMax = 10**5
    n = 0
    xit = []
    
    d0 = -grad(x0)
    rho0 = rechercheDuPas(x0, d0)
    x1 = x0 + rho0 * d0
    n += 1
    
    while ((np.linalg.norm(grad(x0)) > tol) and (n < NiterMax)):
        xit.append(x1)
        x0 = x1
        d0 = -grad(x0)
        rho0 = rechercheDuPas(x0, d0)
        x1 = x0 + rho0 * d0
        n += 1
    
    sol = x1
    
    return sol, xit, n, np.linalg.cond(H(x0))


# -------------------------------------------------------------------
#                   Méthodes de précondtionnement
# -------------------------------------------------------------------


def gradientPrecondtionnePasOptimal(x0, tol):
    
    NiterMax = 10**5
    n = 0
    xit = []
    
    D0 = Dj(x0)
    d0 = np.linalg.solve(D0, -grad(x0))
    rho0 = rechercheDuPas(x0, d0)
    x1 = x0 + rho0 * d0
    n += 1
    
    while ((np.linalg.norm(grad(x0)) > tol) and (n < NiterMax)):
        xit.append(x1)
        x0 = x1
        D0 = Dj(x0)
        d0 = np.linalg.solve(D0, -grad(x0))
        rho0 = rechercheDuPas(x0, d0)
        x1 = x0 + rho0 * d0
        n += 1
    
    sol = x1
    
    return sol, xit, n, np.linalg.cond(H(x0))


def methodeNewton(x0, tol):
    
    NiterMax = 10**5
    n = 0
    xit = []
    
    D0 = H(x0)
    d0 = np.linalg.solve(D0, -grad(x0))
    rho0 = rechercheDuPas(x0, d0)
    x1 = x0 + rho0 * d0
    n += 1
    
    while ((np.linalg.norm(grad(x0)) > tol) and (n < NiterMax)):
        xit.append(x1)
        x0 = x1
        D0 = H(x0)
        d0 = np.linalg.solve(D0, -grad(x0))
        rho0 = rechercheDuPas(x0, d0)
        x1 = x0 + rho0 * d0
        n += 1
    
    sol = x1
    
    return sol, xit, n, np.linalg.cond(H(x0))
    
    
def methodeQuasiNewton(x0, tol):
    
    NiterMax = 10**5
    n = 0
    p = x0.size
    xit = []
    
    D0 = np.eye(p)
    d0 = np.linalg.solve(D0, -grad(x0))
    rho0 = rechercheDuPas(x0, d0)
    
    x1 = x0 + rho0 * d0
    y0 = grad(x1) - grad(x0)

    s0 = rho0 * d0
    D1 = D0 + ((y0 @ y0.T) / (y0.T @ s0)) - (D0 @ s0 @ s0.T @ D0) / (s0.T @ D0 @ s0) 
    n += 1
    
    while ((np.linalg.norm(grad(x0)) > tol) and (n < NiterMax)):
        
        D0 = D1
        x0 = x1
        
        d0 = np.linalg.solve(D0, -grad(x0))
        rho0 = rechercheDuPas(x0, d0)
        s0 = rho0 * d0
        
        x1 = x0 + s0
        y0 = grad(x1) - grad(x0)
        
        D1 = D0 + ((y0 @ y0.T) / (y0.T @ s0)) - (D0 @ s0 @ s0.T @ D0) / (s0.T @ D0 @ s0) 
        n += 1
    
    sol = x1
    
    return sol, xit, n, np.linalg.cond(H(x0))
    
    

if __name__ == '__main__':
    
    x0 = np.array([[4, -2]]).T
        
    sol, xit, n, condH = gradientPasFixe(x0, 10**(-3), 10**(-4))
    print(f"""
          
        -------------- Méthode du gradient à pas fixe --------------
            Le point critique de f(x,y) est :
                        x* = [{sol[0]}
                              {sol[1]}]        
            Le nombre d'itérations est de {n}
            Conditionnement de la Hessienne : {condH}
        ------------------------------------------------------------""")

    sol, xit, n, condH= gradientPasOptimal(x0, 10**(-4))
    print(f"""
        -------------- Méthode du gradient à pas optimal ------------
            Le point critique de f(x,y) est :
                        x* = [{sol[0]}
                              {sol[1]}]        
            Le nombre d'itérations est de {n}
            Conditionnement de la Hessienne : {condH}
        -------------------------------------------------------------""")
    
    sol, xit, n, condH = gradientPrecondtionnePasOptimal(x0, 10**(-4))
    print(f"""
        ----- Méthode du gradient préconditionné à pas optimal ------
            Le point critique de f(x,y) est :
                        x* = [{sol[0]}
                              {sol[1]}]       
            Le nombre d'itérations est de {n}
            Conditionnement de la Hessienne : {condH}
        -------------------------------------------------------------""")
    
    sol, xit, n, condH = methodeNewton(x0, 10**(-4))
    print(f"""
        -------------------- Méthode de Newton ---------------------
            Le point critique de f(x,y) est :
                        x* = [{sol[0]}
                              {sol[1]}]        
            Le nombre d'itérations est de {n}
            Conditionnement de la Hessienne : {condH}
        -------------------------------------------------------------""")
    
    sol, xit, n, condH = methodeQuasiNewton(x0, 10**(-4))
    print(f"""
        ---------------- Méthode quasi-Newton BFGS ------------------
            Le point critique de f(x,y) est :
                        x* = [{sol[0]}
                              {sol[1]}]       
            Le nombre d'itérations est de {n}
            Conditionnement de la Hessienne : {condH}
        -------------------------------------------------------------""")

    
    
    
    
    
