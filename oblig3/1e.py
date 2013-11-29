from dolfin import *
import numpy as np
from diffusion import DiffusionProblem, DiffusionSolver

class TestProblem(DiffusionProblem):
    def __init__(self, rho):
        self.rho = rho 

    def initial_condition(self):
        return Constant(0.0)

    def alpha(self):
        return lambda u: 1 + u**2
        
    def source_term(self):
        code = '''
        - rho*pow(x[0],3)/3.0
        + rho*pow(x[0],2)/2.0
        + 8*pow(t,3)*pow(x[0],7)/9.0
        - 28*pow(t,3)*pow(x[0],6)/9.0
        + 7*pow(t,3)*pow(x[0],5)/2.0
        - 5*pow(t,3)*pow(x[0],4)/4.0
        + 2*t*x[0]-t
        '''
        return Expression(code, rho=self.rho, t=0.0)

T = 0.1
results = []

for h in [0.5, 0.1, 0.05, 0.011, 0.005, 0.001]:
    dt = np.sqrt(h)
    r = int(1./dt)
    
    problem = TestProblem(rho=1.5)
    solver = DiffusionSolver(problem, [r], dt=dt, deg=1)
    solver.solve(T, plot_realtime=False)

    u = solver.get_solution()
    exact = Expression('t*x[0]*x[0]*(0.5 - x[0]/3.)', t=T)
    u_e = project(exact, solver.V)
    e = u_e.vector().array() - u.vector().array()
    #E = np.sqrt(np.sum(e**2)/u.vector().array().size)
    E = max(e**2)

    results.append("h: %3.1e \t\t E: %6.3e \t\t E/h: %6.3e" % (h, E, E/h))

print "Error for different h"
print "\n".join(results)

