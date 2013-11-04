from wave2D import *

class SubseaHillProblem(WaveProblem):
    def __init__(self, I_0, I_a, I_s, B_0, B_a, B_mx, B_s, B_my, bscale, b, g, hill):
        self.I_0, self.I_a, self.I_s = I_0, I_a, I_s
        self.B_0, self.B_a, self.B_mx, self.B_s, self.B_my = B_0, B_a, B_mx, B_s, B_my
        self.bscale, self.b = bscale, b
        self.g = g
        
        if hill == "flat":
            self.B = self.flat

        if hill == "gaussian":
            self.B = self.gaussian_hill

        if hill == "hat":
            self.B = self.cosine_hat

        if hill == "box":
            self.B = self.box

    def I(self, x, y):
        I_0, I_a, I_s =  self.I_0, self.I_a, self.I_s
        return I_0 + I_a*np.exp(-(x/I_s)**2)

    def V(self, x, y):
        return 0.0

    def q(self, x, y):
        return self.g*self.H(x,y)

    def H(self, x, y):
        return self.B_0 - self.B(x,y)

    def B(self, x, y):
        raise NotImplementedError

    def gaussian_hill(self, x, y):
        B_0, B_a, B_mx, B_s, B_my, b = self.B_0, self.B_a, self.B_mx,self.B_s, self.B_my, self.bscale
        return B_0 + B_a*np.exp(-((x-B_mx)/B_s)**2 - ((y-B_my)/(b*B_s))**2)

    def cosine_hat(self, x, y):
        B_0, B_a, B_mx, B_s, B_my = self.B_0, self.B_a, self.B_mx,self.B_s, self.B_my
        
        if np.sqrt(x**2 + y**2) <= B_s:
            return B_0 + B_a*np.cos(np.pi*(x-B_mx)/(2*B_s))*np.cos(np.pi*(y-B_my)/(2*B_s))
        else:
            return B_0

    def box(self, x, y):
        B_0, B_a, B_mx, B_s, B_my, b = self.B_0, self.B_a, self.B_mx,self.B_s, self.B_my, self.bscale

        if B_mx-B_s<=x<=B_mx+B_s and B_my-b*B_s<=y<=B_my+b*B_s:
            return B_0 + B_a
        else:
            return B_0

    def flat(self, x, y):
        return 0.0



I_0 = 0
I_a = 0.5
I_s = 2.0
B_0 = 2.0
B_a = 2.5
B_mx = 1.5
B_my = 1.5 
B_s = 0.5
bscale = 1.0
b = 0.0
g = 9.81
hill = "flat"
problem = SubseaHillProblem(I_0, I_a, I_s, B_0, B_a, B_mx, B_s, B_my, bscale, b, g, hill)

Lx = 3.0
Ly = 3.0
Nx = 200
Ny = 200
dt = 0.001
T = 1
solver = WaveSolver(problem, Lx, Ly, Nx, Ny, dt, T, version="vectorized")


plot_solutions(solver, show=False, save="test", stride=[15,15])


