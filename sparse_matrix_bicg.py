import numpy as np
import numpy.linalg as linalg
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

def bicg(A, b, x0, niter, epsilon):
    conv_history = -np.ones(niter+1, dtype=np.double)
    iter = 0
    r = b - prodAx(A[0],A[1],x0)
    rs= r.copy()
    pk = r.copy()
    uk = r.copy()
    sqnrmr0 = np.dot(r,r)
    sqnrmr  = sqnrmr0
    conv_history[0] = 1
    x = x0.copy()
    dotrrs = np.dot(r, rs) 
    while (iter < niter) and (sqnrmr > epsilon * epsilon * sqnrmr0) :
        Apk = prodAx(A[0],A[1],pk)
        alpha = dotrrs/np.dot(Apk,rs)
        qk = uk - alpha*Apk
        aukqk = alpha * (uk+qk)
        x = x + aukqk
        r = r - prodAx(A[0],A[1],aukqk)        
        new_dotrrs = np.dot(r, rs)
        beta = new_dotrrs/dotrrs
        dotrrs = new_dotrrs
        uk = r + beta * qk
        pk = uk + beta*(qk + beta*pk)
        sqnrmr = np.dot(r,r)
        iter = iter + 1
        conv_history[iter] = sqrt(sqnrmr/sqnrmr0)

    return x, iter, conv_history

def display_historic(historic):
    import pylab
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(num="BiCG")
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
sol, niter, history = bicg(A, b, x0, 500, 1.E-6)
end = time.time()

r = prodAx(A[0],A[1],sol) - b
print(f"norme effective : {linalg.norm(r)/linalg.norm(b)}")
print(f"Résolution du système en {end-beg} secondes")
display_historic(history)