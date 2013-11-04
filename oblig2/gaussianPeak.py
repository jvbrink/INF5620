from wave2D import *

def gaussian_peak():
    '''Illustrating a simple use of the WaveProblem and WaveSolver classes.'''
    class GaussianWithDamping(WaveProblem):
        def __init__(self, Lx, Ly, b):
            self.Lx, self.Ly, self.b = Lx, Ly, b

        def q(self,x,y):
            return 20.0

        def I(self,x,y):
            Lx, Ly = self.Lx, self.Ly
            return 2*np.exp(-(x-0.5*Lx)**2 - (y-0.5*Ly)**2/2)

    Lx = 5
    Ly = 10
    Nx = 40
    Ny = 80
    dt = 0.01
    T = 2.0
    b = 2.0
    version="vectorized"
    problem = GaussianWithDamping(Lx, Ly, b)
    solver = WaveSolver(problem, Lx, Ly, Nx, Ny, dt, T, version=version)
    plotter = WavePlotter(solver)
    
    plotter.solve_and_plot(save='test', zlim=2)
    

if __name__ == '__main__':
    gaussian_peak()