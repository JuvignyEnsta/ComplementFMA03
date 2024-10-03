from math import exp 

def my_exp(x, n):
    fact = 1
    res  = 1
    for i in range(1,n):
        fact *= i
        res += (x**i)/fact
    return res 

my_x = my_exp(-10, 30)
print(f"naïve e^{-10} = {my_x} contre {exp(-10)}")

my_y = my_exp(+10, 30)
my_x = 1./my_y

print(f"preci e^{-10} = {my_x} contre {exp(-10)}")