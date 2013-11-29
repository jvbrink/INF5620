from dolfin import *
import numpy as np
from diffusion import DiffusionProblem, DiffusionSolver

class GaussianDiffusion(DiffusionProblem):
    def __init__(self, rho=0.1, beta=0.1, sigma=0.1):
        self.rho = rho
        self.beta = beta
        self.sigma = sigma

    def initial_condition(self):
        code = 'exp(-1/(2*pow(sigma,2))*(pow(x[0],2)+pow(x[1],2)))'
        return Expression(code, sigma=self.sigma)

    def alpha(self):
        beta = self.beta
        return lambda u: 1 + beta*u**2
    
    def source_term(self):
        return Constant(0.0)

class GaussianSolver(DiffusionSolver):
    def plot_arguments(self):
        args = DiffusionSolver.plot_arguments(self)
        args['mode'] = 'auto'
        args['interactive'] = True
        args['range_min'] = 0.
        args['range_max'] = 1.
        return args

T = 5.0
problem = GaussianDiffusion(rho=15, beta=0.7, sigma=0.5)
solver = GaussianSolver(problem, [40, 40], dt=0.01, deg=1)
solver.solve(T, plot_realtime=True)
