from math import log
def naive_integral(n):
    I = log(11/10)
    for i in range(1,n+1):
        I = 1/i -10*I
    return I

def preci_integral(n):
    I = 1/(11*(n+11))
    for i in range(n+10, n, -1):
        I = (1/i-I)/10
    return I 


for i in range(1,37):
    print(f"intégrale naïve({i}) :\t {naive_integral(i)}")
    print(f"intégrale précise({i}):\t {preci_integral(i)}")
