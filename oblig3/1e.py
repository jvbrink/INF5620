from dolfin import *
import numpy as np
from sympy import *
from diffusion import DiffusionProblem, DiffusionSolver

class TestProblem(DiffusionProblem):
    def initial_condition(self):
        return Constant(0.0)

    def alpha(self):
        def a(u):
                return 1 + u**2
        return a
        
    def source_term(self):
        return Expression('-rho*pow(x[0],3)/3.0+rho*pow(x[0],2)/2.0+8*pow(t,3)*pow(x[0],7)/9.0-28*pow(t,3)*pow(x[0],6)/9.0+7*pow(t,3)*pow(x[0],5)/2.0-5*pow(t,3)*pow(x[0],4)/4.0+2*t*x[0]-t', rho=1.5, t=0.0)
        

class TestSolver(DiffusionSolver):
    def plot_arguments(self):
        args = DiffusionSolver.plot_arguments(self)
        args['mode'] = 'auto'
        args['range_min'] = -1.
        args['range_max'] = 1.
        return args


problem = TestProblem(rho=2.0)
solver = TestSolver(problem, [10], dt=0.01, deg=1)
T = 1.0
solver.solve(T, plot_realtime=False)


#     u = solver.up
#     exact_solution = Expression('exp(-pi*pi*t)*cos(pi*x[0])', t=T)
#     u_e = project(exact_solution, solver.V)

#     e = u_e.vector().array() - u.vector().array()
#     E = max(e**2)

#     results.append("h: %3.1e \t\t E: %6.3e" % (h, E))

# print "\n".join(results)