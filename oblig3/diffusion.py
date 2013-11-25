from dolfin import *
import numpy as np

class DiffusionProblem:
    """
    Class for defining a nonlinear diffusion model.
    """
    def __init__(self, rho=1.0):
        self.rho = rho

    def initial_condition(self):
        # Returns a Dolfin Expression-object
        raise NotImplementedError

    def f(self):
        # Returns a Dolfin Expression-object
        raise NotImplementedError
    
    def alpha(self, u):
        # Returns a Dolfin Expression-object
        raise NotImplementedError


class DiffusionSolver:
    """
    Class for solving a DiffusionProblem on the unit
    hypercube.
    """

    def __init__(self, problem, res, dt=0.01, deg=1):
        domain_type = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
        mesh = domain_type[len(res)-1](*res)

        V = FunctionSpace(mesh, 'Lagrange', deg)
        u  = TrialFunction(V)
        v  = TestFunction(V)

        # Get the initial condition Expression object
        I = problem.initial_condition()
        # Project the expression onto the mesh
        up = project(I, V)

        # Get the alpha Expression-object and the source term
        alpha = problem.alpha()
        self.f = problem.source_term()
        self.fp = problem.source_term()

        # Define variational form
        rho = problem.rho
        self.a = (2*rho/dt*u*v + alpha(up)*inner(grad(u), grad(v)))*dx
        self.L = (2*rho/dt*up*v - alpha(up)*inner(grad(up), grad(v)) \
                    + self.f*v + self.fp*v)*dx
        
        self.up = up
        self.v = v
        self.V = V
        self.dt = dt
        self.n = 0

    def step(self):
        # Update source terms
        self.f.t = (self.n+1)*self.dt
        self.fp.t = self.n*self.dt 

        # Picard iteration
        u = Function(self.V)
        solve(self.a == self.L, u)

        # Update up and n
        self.up.assign(u)
        self.n += 1

    def solve(self, T, plot_realtime=False):
        '''Solve from current time to time T'''
        while self.n*self.dt < T:
            # Solve for one more time step
            self.step()
            
            # Plot solution at current time-step
            if plot_realtime:
                self.plot_solution()


    def plot_solution(self):
        """Plot current solution"""
        plot(self.up, **self.plot_arguments())

    def plot_arguments(self):
        """Return dictionary with arguments to plot command in solve."""
        args = {'interactive' : True}
        args = {"mode"           : "auto",
              "interactive"    : True,
              "wireframe"      : False,
              "rescale"        : True,
              "tile_windows"   : True}
        return args




    


