from wave2D_ghost import *

def test_separabel(c=np.pi):
    """
    Test the solver using a separabel exact solution on the form
        u_e = X(x)Y(y)T(t)
    where X and Y are cubic polynomials, and T is a quadratic polynomial.
    With a constant wave velocity q, and no damping, b = 0.
    Implements a nose test.
    """
    import nose.tools as nt

    class CubicSolution(Problem):
        def __init__(self, A, B, D, X, Y, T, dx, dy, Lx, Ly, q_const=2.0):
            self.A, self.B, self.D = A, B, D
            self.dx, self.dy, self.Lx, self.Ly = dx, dy, Lx, Ly
            self.X, self.Y, self.T = X, Y, T
            self.q_const = q_const
            self.b = 0

        def V(self, x, y):
            return self.X(x)*self.Y(y)

        def f(self, x, y, t):
            A, B, D = self.A, self.B, self.D
            dx, dy, Lx, Ly = self.dx, self.dy, self.Lx, self.Ly
            X, Y, T = self.X, self.Y, self.T
            q = self.q_const

            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                # Array evaluation
                # Evaluate all mesh points
                f_a = - q*Y(y[np.newaxis,:])*T(t)*A*(6*x[:,np.newaxis]-3*Lx) \
                - q*X(x[:,np.newaxis])*T(t)*B*(6*y[np.newaxis,:]-3*Ly)
                # Add extra contributions at boundaries
                f_a[0,:]  += 2*q*Y(y[:])*T(t)*A*dx
                f_a[-1,:] += -f_a[0,:]
                f_a[:,0]  += 2*q*X(x[:])*T(t)*B*dy
                f_a[:,-1] += -f_a[:,0]
                return f_a
            
            else:
                # Pointwise evaluation
                f_v = -q*Y(y)*T(t)*A(6*x - 3*Lx) \
                                    - q*X(x)*T(t)*B(6*y - 3*Ly)
                # Add extra contributions at boundaries
                tol = 1e-14
                if abs(x-Lx) < tol:
                    f_v += 2*q*Y(y)*T(t)*A*dx
                if abs(x) < tol:
                    f_v -= 2*q*Y(y)*T(t)*A*dx
                if abs(y) < tol:
                    f_v += 2*q*X(x)*T(t)*B*dy
                if abs(y-Ly) < tol:
                    f_v -= 2*q*X(x)*T(t)*B*dy
                return f_v

    Lx = 2
    Ly = 4
    Nx = 40
    Ny = 80
    dx = Lx/float(Nx)
    dy = Ly/float(Ny)
    dt = 0.01
    endT = 1
    q = 2.0
    b = 0.0
    A = 2.
    B = 2.
    D = 1.
    Cx = 0.
    Cy = 0.

    X = lambda x: A*x**3 - 1.5*A*Lx*x**2 + Cx
    Y = lambda y: B*y**3 - 1.5*B*Ly*y**2 + Cy
    T = lambda t: D*t

    problem = CubicSolution(A, B, D, X, Y, T, dx, dy, Lx, Ly)

    print "Verifying solver for a cubic solution."
    for v in ["vectorized"]:
        solver = Solver(problem, Lx, Ly, Nx, Ny, dt, endT, version=v)
        x, y = solver.get_mesh()

        while solver.n < solver.Nt:
            solver.advance()
            x, y = solver.get_mesh()        
            u_e = X(x[:,np.newaxis])*Y(y[np.newaxis,:])*T(solver.t[solver.n])
            u = solver.get_solution()

            plot_solution(x, y, u.T)
            plot_solution(x, y, u_e.T)

            print abs(u-u_e).max()

        '''
        solver.solve()
        u = solver.get_solution()
        diff = abs(u-u_e).max()
        print "%10s: abs(u-u_e).max() = %e" % (v, diff)
        nt.assert_almost_equal(diff, 0, places=12)
        '''


if __name__ == "__main__":
    test_separabel()