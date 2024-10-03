import numpy as np
import numpy.linalg as linalg
from math import sqrt

def matrix_laplacian(n : int):
    n2 = (n-2)*(n-2)
    A = np.zeros( (n2,n2), dtype=np.double )
    ## Diagonale :
    np.fill_diagonal(A, 4.);
    ## Sous-diagonale :
    np.fill_diagonal(A[1:,:], -1.)
    for i in range(n-2):
        A[i*(n-2) , i*(n-2)-1] = 0
    np.fill_diagonal(A[:,1:], -1.)
    for i in range(n-2):
        A[i*(n-2)-1, i*(n-2)] = 0
    np.fill_diagonal(A[:,n-2:], -1)
    np.fill_diagonal(A[n-2:,:], -1)
    return A

def compute_rhs(n : int, h : float ):
    nm2 = n - 2
    b = np.zeros(nm2*nm2, dtype=np.double)
    for i in range(nm2):
        b[i] = h*h
    return b

def gradient_conjugue(A, b, x0, niter, epsilon):
    conv_history = -np.ones(niter+1, dtype=np.double)
    iter = 0
    r = b - A@x0
    pk = r.copy()
    sqnrmr0 = np.dot(r,r)
    sqnrmr  = sqnrmr0
    conv_history[0] = 1
    x = x0.copy()
    while (iter < niter) and (sqnrmr > epsilon * epsilon * sqnrmr0) :
        Apk = A@pk
        alpha = sqnrmr/np.dot(Apk,pk)
        x += alpha * pk
        r -= alpha * Apk
        new_sqnrm = np.dot(r,r)
        beta = new_sqnrm/sqnrmr
        sqnrmr = new_sqnrm
        pk = r + beta * pk
        iter = iter + 1
        conv_history[iter] = sqrt(sqnrmr/sqnrmr0)

    return x, iter, conv_history

def display_historic(historic):
    import pylab
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(num="Conjugate gradient Convergence")
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
A = matrix_laplacian(dim)
print(A)
b = compute_rhs(dim, 1./(dim-1.))
x0 = np.zeros((dim-2)*(dim-2), dtype=np.double)

sol, niter, history = gradient_conjugue(A, b, x0, 500, 1.E-6)

print(sol)
print(niter)
print(history)
display_historic(history)