from wave2D_ghost import *

def test_cubic(c=np.pi):
    """
    Test the solver using an exact solution on the form
        u_e = X(x)Y(y)T(t)
    where X and Y are cubic polynomials, and T is a quadratic polynomial.
    With a constant wave velocity q, and no damping, b = 0.
    Implements a nose test.
    """
    import nose.tools as nt

    class CubicSolution(Problem):
        def __init__(self, ax, ay, at, X, Y, T, Lx, Ly):
            self.ax, self.ay, self.at = ax, ay, at
            self.X, self.Y = X, Y
            self.Lx, self.Ly = Lx, Ly
            self.q_const = 2.0
            self.b = 0

        def f(self, x, y, t):
            ax, ay, at = self.ax, self.ay, self.at
            X, Y = self.X, self.Y
            Lx, Ly = self.Lx, self.Ly
            q = self.q_const

            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                return 2*at*X(x[:,np.newaxis])*Y(y[np.newaxis,:]) \
                - q*(6*ax*x[:,np.newaxis] - 3*ax*Lx)*Y(y[np.newaxis,:])*T(t) \
                - q*(6*ay*y[np.newaxis,:] - 3*ay*Ly)*X(x[:,np.newaxis])*T(t)                
            else:
                return 2*at*X(x)*Y(y) - q*(6*ax*x - 3*ax*Lx)*Y(y)*T(t) \
                                      - q*(6*ay*y - 3*ay*Ly)*X(x)*T(t)
    Lx = 2
    Ly = 4
    Nx = 40
    Ny = 80
    dt = 0.01
    endT = 1
    q = 2.0
    b = 0.0
    ax = 2.3
    ay = 3.7
    at = -1.1
    cx = 17.2
    cy = -3.9

    X = lambda x: ax*x**3 - 1.5*ax*Lx*x**2 + cx
    Y = lambda y: ay*y**3 - 1.5*ay*Ly*y**2 + cy
    T = lambda t: at*t**2

    problem = CubicSolution(ax, ay, at, X, Y, T, Lx, Ly)

    print "Verifying solver for a cubic solution."
    for v in ["vectorized"]:
        solver = Solver(problem, Lx, Ly, Nx, Ny, dt, endT, version=v)

        x, y = solver.get_mesh()        
        solver.advance()
        u_e = X(x[:,np.newaxis])*Y(y[np.newaxis,:])*T(solver.t[solver.n])
        plot_error(solver, u_e, save='cubic_error')
    
        
        '''
        solver.solve()
        u = solver.get_solution()
        diff = abs(u-u_e).max()
        print "%10s: abs(u-u_e).max() = %e" % (v, diff)
        nt.assert_almost_equal(diff, 0, places=12)
        '''


if __name__ == "__main__":
    test_cubic()