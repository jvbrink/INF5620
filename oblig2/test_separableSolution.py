from wave2D import *

def test_separable(c=np.pi):
    """
    Verify the solver with a separable exact solution
        u_e = X(x)Y(y)T(t)
    where
        X = Ax^3 - (3/2)*A*Lx*x**2
        Y = By^3 - (3/2)*B*Ly*y**2
        T = D*t
    Uses a constant wave velocity q, and no damping, b = 0.
    """
    import nose.tools as nt

    class CubicSolution(WaveProblem):
        def __init__(self, A, B, D, dx, dy, Lx, Ly, q_const=2.0):
            self.A, self.B, self.D = A, B, D
            self.dx, self.dy = dx, dy 
            self.Lx, self.Ly = Lx, Ly
            self.q_const = q_const
            self.b = 0

        def X(self, x):
            A, Lx = self.A, self.Lx
            return A*x**3 - 1.5*A*Lx*x**2

        def Y(self, y):
            B, Ly = self.B, self.Ly
            return B*y**3 - 1.5*B*Ly*y**2

        def T(self, t):
            D = self.D
            return D*t

        def u_e(self, x, y, t):
            X, Y, T = self.X, self.Y, self.T
            return X(x)*Y(y)*T(t)

        def V(self, x, y):
            X, Y = self.X, self.Y
            return D*X(x)*Y(y)

        def I(self,x,y):
            return self.u_e(x,y,0)

        def f(self, x, y, t):
            A, B, D = self.A, self.B, self.D
            X, Y, T = self.X, self.Y, self.T
            Lx, Ly = self.Lx, self.Ly
            dx, dy = self.dx, self.dy
            q = self.q_const
            

            # Vector evaluated
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                # All mesh points
                fx = A*(6*x[:,np.newaxis] - 3*Lx)*Y(y[np.newaxis,:])
                fy = B*(6*y[np.newaxis,:] - 3*Ly)*X(x[:,np.newaxis])
                f  = -q*(fx + fy) * T(t)
                # Add extra contributions at boundaries
                f[0,:]  -= 2*A*dx*q*Y(y[:])*T(t)
                f[-1,:] += 2*A*dx*q*Y(y[:])*T(t)
                f[:,0]  -= 2*B*dy*q*X(x[:])*T(t)
                f[:,-1] += 2*B*dy*q*X(x[:])*T(t)

                # print "time ", t
                # print f

            # Pointwise evaluated
            else:
                fx = A*(6*x - 3*Lx)*Y(y)
                fy = B*(6*y - 3*Ly)*X(x)
                f = -q*(fx + fy) * T(t)
                # Add extra contributions at boundaries
                tol = 1e-14
                if abs(x) < tol:
                    f -= 2*A*dx*q*Y(y)*T(t)
                if abs(x-Lx) < tol:
                    f += 2*A*dx*q*Y(y)*T(t)
                if abs(y) < tol:
                    f -= 2*B*dy*q*X(x)*T(t)
                if abs(y-Ly) < tol:
                    f += 2*B*dy*q*X(x)*T(t)
            
            return f

    Lx = 0.3
    Ly = 0.3
    Nx = 3
    Ny = 3
    dx = Lx/float(Nx)
    dy = Ly/float(Ny)
    dt = 0.01
    T = 10
    q = 1.2
    b = 0.0
    A = 2.
    B = 2.
    D = 1.

    problem = CubicSolution(A, B, D, dx, dy, Lx, Ly, q_const=q)

    print "Verifying solver for a separable solution."
    for v in ["scalar", "vectorized"]:
        solver = WaveSolver(problem, Lx, Ly, Nx, Ny, dt, T, version=v)
        x, y = solver.get_mesh()

        u = solver.get_solution()

        while solver.n < solver.Nt:                
            solver.advance()
            t = solver.t[solver.n]
            
        u_e = problem.u_e(x[:,np.newaxis],y[np.newaxis,:],t)
        u = solver.get_solution()
        diff = abs(u-u_e).max()
        print "%10s: abs(u-u_e).max() = %e" % (v, diff)
        nt.assert_almost_equal(diff, 0, places=12)
    
if __name__ == "__main__":
    test_separable()
