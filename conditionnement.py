import numpy.linalg as nalg 
import scipy.linalg as salg
import numpy as np

A = np.array([[10, 7, 8, 7], 
               [7,5,6,5],
               [8,6,10,9],
               [7,5,9,10]])
b = np.array([32,23, 33,31])

dA = np.array([[0, 0, 0.1, 0.2],
               [0.08, 0.04, 0, 0],
               [0, -0.02, -0.11, 0],
               [-0.01, -0.01, 0, -0.02]])

db = np.array([0.01, -0.01, 0.01, -0.01])

x = salg.solve(A, b)
y = salg.solve(A,b+db)
z = salg.solve(A+dA,b)

print(f"x = {x}, y = {y}, z = {z}")

