from dolfin import *
import numpy as np
from diffusion import DiffusionProblem, DiffusionSolver

class TestProblem(DiffusionProblem):
    def alpha(self):
        return Constant(1.0)
    
    def initial_condition(self):
        return Expression('cos(pi*x[0])')

    def source_term(self):
        return Constant(0.0)

class TestSolver(DiffusionSolver):
    def plot_arguments(self):
        args = DiffusionSolver.plot_arguments(self)
        args['mode'] = 'auto'
        args['range_min'] = -1.
        args['range_max'] = 1.
        return args


problem = TestProblem(rho=2.0)

T = 3.0
results = []

for h in [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
    dt = np.sqrt(h)
    r = int(1./dt)

    solver = TestSolver(problem, [r, r], dt=dt, deg=1)
    solver.solve(T, plot_realtime=False)

    u = solver.up
    exact_solution = Expression('exp(-pi*pi*t)*cos(pi*x[0])', t=T)
    u_e = project(exact_solution, solver.V)

    e = u_e.vector().array() - u.vector().array()
    #E = np.sqrt(np.sum(e**2)/u.vector().array().size)
    E = max(e**2)

    results.append("h: %3.1e \t\t E: %6.3e \t\t E/h: %6.3e" % (h, E, E/h))


dtp = 1
solver = TestSolver(problem, [10, 10], dt=dtp, deg=1)
solver.solve(T, plot_realtime=False)
u = solver.up
exact_solution = Expression('exp(-pi*pi*t)*cos(pi*x[0])', t=T)
u_e = project(exact_solution, solver.V)
e = u_e.vector().array() - u.vector().array()
Ep = max(e**2)

convergence = []
for dt in [1e-1, 1e-2, 1e-3]:
    solver = TestSolver(problem, [10, 10], dt=dt, deg=1)
    solver.solve(T, plot_realtime=False)

    u = solver.up
    e = u_e.vector().array() - u.vector().array()
    #E = max(e**2)
    E = np.sqrt(np.sum(e**2)/u.vector().array().size)

    r = ln(Ep/E)/ln(dtp/dt)
    Ep = E
    dtp = dt
    
    convergence.append("dt: %3.1e \t\t r: %2.2e" % (dt, r))


print "Error for different h"
print "\n".join(results)

print "Testing convergence rate for dt"
print "\n".join(convergence)
    
