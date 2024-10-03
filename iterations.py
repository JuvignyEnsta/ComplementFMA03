from math import log
def iter(un):
    return abs(log(un))

u = 2
v = 2+1.E-10
w = 2-1.E-10

print(f"u0 : \t{u}, v0 : \t{v}, w0 :\t{w}")

for i in range(30):
    u = iter(u)
    v = iter(v)
    w = iter(w)
    print(f"u{i+1} : \t{u}, v{i+1} : \t{v}, w{i+1} :\t{w}")
