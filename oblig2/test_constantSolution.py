from wave2D import *

def test_constant(c=np.pi):
    """
    Test the solver using constant solution, u_e=C. No source term
    is then needed. Implements a nose test.
    """
    import nose.tools as nt

    class ConstantSolution(WaveProblem):
        def __init__(self, b, c=c, q_const=1.0):
            self.b, self.c, self.q_const = b, c, q_const
        
        def I(self, x, y):
            return c

    Lx = 2
    Ly = 4
    Nx = 40 
    Ny = 80
    dt = 0.01
    T = 0.1
    q = 2.0
    b = 3.0
    f = 0
    problem = ConstantSolution(b, c=c, q_const=q)

    print "Verifying solver for a constant solution u_e=const."
    for v in ["scalar", "vectorized"]:
        solver = WaveSolver(problem, Lx, Ly, Nx, Ny, dt, T, version=v)
        u_e = np.zeros(solver.get_solution().shape)
        u_e.fill(c)
        solver.solve()
        u = solver.get_solution()
        diff = abs(u-u_e).max()
        print "%10s: abs(u-u_e).max() = %e" % (v, diff)
        nt.assert_almost_equal(diff, 0, places=12)

if __name__ == "__main__":
    test_constant()