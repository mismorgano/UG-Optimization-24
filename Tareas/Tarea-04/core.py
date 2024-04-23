import numpy as np

from sympy.utilities.lambdify import lambdify
from sympy import Matrix, symbols, exp
from sympy.utilities.iterables import flatten


def gradient(f, var):
    partials = [f.diff(x) for x in var]
    return flatten(Matrix(partials))

def hessian(f, var):
    partials = [[f.diff(x, y) for y in var] for x in var]
    return Matrix(partials)

def build_f_gradient_hessian(f, var):
    dim = len(var)
    g = lambdify(var, gradient(f, var))
    h = lambdify(var, hessian(f, var))

    def grad(x):
        assert len(x) == dim, f"{len(x)}, {dim}"
        return np.array(g(*x))
    def hess(x):
        assert len(x) == dim, f"{len(x)}, {dim}"
        return h(*x)
    f = lambdify(var, f)
    # def nf(x):
    #     return f(*x)
    # como implementar una funcion para que acepte np.array

    return f, grad, hess

# Utilizamos sympy para hacer la diferenciación simbolica


# RR^2 -> RR functions
x, y = symbols("x y")

# Himmelblau function
himmelblau = (x**2 + y -  11)**2 + (x + y**2 -7)**2
himme, himme_grad, himme_hess = build_f_gradient_hessian(himmelblau, [x, y])

# beale function
beale = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
beale, beale_grad, beale_hess = build_f_gradient_hessian(beale, [x, y])



# rosenbrok function
def rosenbrock(n=2):
    # n = len(var)
    y = symbols("0")
    x = symbols(f"x1:{n+1}")
    # return Sum(100*(x[i+1] -x[i]**2)**2 + (1-x[i])**2, (i, 1, n-1))
    for i in range(n-1):
        y += 100*(x[i+1] -x[i]**2)**2 + (1-x[i])**2
    return y, x

rosen, rosen_grad, rosen_hess = build_f_gradient_hessian(*rosenbrock())

# hartmann
a = Matrix([1.0, 1.2, 3.0, 3.2])
A = Matrix([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], 
     [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])

P = 1e-4*Matrix([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], 
     [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
# print(P)
def hartmann():
    x = symbols(f"x1:{7}")
    y = symbols("2.58")
    for i in range(4):
        z = symbols("0")
        for j in range(6):
            # print( (P[i, j]/10e4 - x[j])**2, A[i, j], P[i, j])
            z -= A[i, j]*(x[j] - P[i, j])**2
        y += a[i]*exp(z)
    return -y/symbols("1.94"), x

hart, hart_grad, hart_hess = build_f_gradient_hessian(*hartmann())


# functions 


