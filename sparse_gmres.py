import numpy as np
import scipy.linalg as linalg
from math import sqrt
import time

def compute_lhs(n : int):
    n2 = (n-2)*(n-2)
    A = np.zeros( 5*n2, dtype=np.double )
    indices = np.zeros(5, dtype=np.int32);
    ## Diagonale :
    A[:n2] = 4.
    indices[0] = 0
    A[n2:2*n2-1] = -0.9
    indices[1] = 1
    for i in range(1,n-2):
        A[n2+i*(n-2)] = 0
    A[2*n2:3*n2-1] = -1.1
    indices[2] = -1
    for i in range(1,n-2):
        A[2*n2+i*(n-2)] = 0
    A[3*n2:4*n2-n+2] = -0.9
    indices[3] = n-2
    A[4*n2:5*n2-n+2] = -1.1
    indices[4] = 2-n 
    return (A, indices)

def prodAx(A, indices, u):
    n = u.shape[0]
    n2 = A.shape[0]//5
    fin_ind = n - indices[1]
    v = A[:n]*u
    v[1:] += A[n2:n2+fin_ind]*u[:fin_ind] 
    v[:-1] += A[2*n2:3*n2+indices[2]]*u[-indices[2]:] 
    v[indices[3]:] += A[3*n2:3*n2+n-indices[3]]*u[:n-indices[3]] 
    v[:indices[4]] += A[4*n2:5*n2+indices[4]]*u[-indices[4]:]
    return v

def compute_rhs(n : int, h : float ):
    nm2 = n - 2
    b = np.zeros(nm2*nm2, dtype=np.double)
    for i in range(nm2):
        b[i] = h*h
    return b

def gmres(A, b, x0, niter, epsilon):
    conv_history = -np.ones(niter+1, dtype=np.double)
    conv_history[0] = 1.
    x = x0.copy()
    r0 = b - prodAx(A[0],A[1],x0)
    beta = linalg.norm(r0)
    nrm0 = beta
    res = beta
    V = [r0/beta]
    H = np.zeros((niter+1,niter+1),type(r0[0]))
    H[niter][0] = beta
    Zm = []
    rotCos = []
    rotSin = []
    jH = 0
    while ( res > epsilon*nrm0 ) and (jH < niter) :
        z = V[jH].copy()
        Zm.append(z)
        w = prodAx(A[0],A[1],z)
        # Modified Gram-Schmidt
        for iv in enumerate(V):
#           H[jH][iv[0]] = dotUV(w,iv[1])
            H[jH][iv[0]] = np.dot(iv[1],w)
            w -=  H[jH][iv[0]]*iv[1]
        duv = np.dot(w,w)
        normw = sqrt(duv)
        if normw == 0 :
            print ("Lucky break-down...")
            break
        H[jH][jH+1] = normw
        V.append(w/normw)
        # Apply previous givens rotation to the new column of H :
        for j in range(jH):
            temp = rotCos[j]*H[jH][j] + rotSin[j]*H[jH][j+1]
            H[jH][j+1] = -rotSin[j]*H[jH][j] + rotCos[j]*H[jH][j+1]
            H[jH][j]   = temp
        gam = sqrt(abs(H[jH][jH])**2 + abs(H[jH][jH+1])**2)
        if abs(H[jH][jH]) != 0 :
            rotSin.append(H[jH][jH+1]*abs(H[jH][jH])/(H[jH][jH]*gam))
            rotCos.append(abs(H[jH][jH])/gam)
        elif abs(H[jH][jH+1]) != 0 :
            rotCos.append(0.)
            rotSin.append(H[jH][jH+1]/abs(H[jH][jH+1]))
        else:
            rotCos.append(1.)
            rotSin.append(0.)
        H[niter][jH+1] = -rotSin[jH]*H[niter][jH]
        H[niter][jH]   =  rotCos[jH]*H[niter][jH]
        H[jH][jH]   = rotCos[jH]*H[jH][jH] + rotSin[jH]*H[jH][jH+1]
        H[jH][jH+1] = 0.
        res = abs(H[niter][jH+1])
        jH = jH + 1
        #print ("It. num. %3d => res = %7.4g"%(jH,res/nrm0))
        conv_history[jH] = res/nrm0
    y = H[niter].copy()
    linalg.solve_triangular(a=H[0:jH,0:jH],b=y[0:jH],trans='T',lower=True,unit_diagonal=False,overwrite_b=True,check_finite=False)
    # Compute solution with Zm :
    for z in enumerate(Zm):
        x += y[z[0]]*z[1]
    # update beta, V, ... for outer loop :
    r0 = (np.asarray(b)).copy()
    r0 = r0 - prodAx(A[0],A[1],x)
    #prodAx(x, r0, -1.+0j, 1.+0j)
    duv = np.dot(r0,r0)
    beta = sqrt(duv.real)
    res = beta
    print ('Effective residual : %7.4g'%(res/nrm0))
    return x, jH, conv_history

def display_historic(historic):
    import pylab
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(num="GMRES")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    w = np.where(historic==-1.)[0]
    end = -1
    if w.shape[0] == 0:
        c, = ax.plot(historic, color=cm.hot(0))
    else:
        c, = ax.plot(historic[:w[0]], color=cm.hot(0))
    pylab.show()


dim = 100
A = compute_lhs(dim)
b = compute_rhs(dim, 1./(dim-1.))
x0 = np.zeros((dim-2)*(dim-2), dtype=np.double)

beg = time.time()
sol, niter, history = gmres(A, b, x0, 200, 1.E-6)
end = time.time()

r = prodAx(A[0],A[1],sol) - b
print(f"norme effective : {linalg.norm(r)/linalg.norm(b)}")
print(f"Résolution du système en {end-beg} secondes")


display_historic(history)