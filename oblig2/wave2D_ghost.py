#!/usr/bin/env python

import numpy as np
from scitools.std import *

class Solver():
	"""
	A solver for solving a standard 2D linear wave equation
	with damping:

	u_tt + b*u_t = (q(x,y)*u_x)_x + (q(x,y)u_y)_y + f(x,y,t)

	with boundary condition: 	u_n = 0

	and  initial conditions: 	u(x,y,0) = I(x,y)
								u_t(x,y,0) = V(x,y)
	"""

	def __init__(self, I, V, q, Lx, Ly, Nx, Ny, dt, T, b,
								 f=None, version="scalar"):
		# Create the meshpoints with ghost points
		dx = Lx/float(Nx)
		dy = Ly/float(Ny)
		x = np.linspace(-dx, Lx+dx, Nx+3) # x-dir
		y = np.linspace(-dy, Ly+dy, Ny+3) # x-dir
		Ix = range(1,x.size-1)
		Iy = range(1,y.size-1)
		
		Nt = int(round(T/float(dt)))
		t = np.linspace(0, Nt*dt, Nt+1) # time
		hx = (dt/dx)**2
		hy = (dt/dx)**2
		
    	# Set up initial conditions
		if type(I)==np.ndarray:
			u = I
		else:
			# Make sure I is callable
			I = I if I else (lambda x, y: 0)
			# Evaluate I in mesh points
			u = np.zeros((x.size, y.size))
			for i in range(x.size):
				for j in range(y.size):
					u[i,j] = I(x[i],y[j])
		# Set values of ghost cells
		u[0, :]  = u[2, :]
		u[-1, :] = u[-3, :]
		u[:, 0]  = u[:, 2]
		u[:, -1] = u[:, -3]

		if type(V) == np.ndarray:
			v = V
		else:
			# Make sure V is callable
			V = V if V else (lambda x, y: 0)
			# Evaluate V in mesh points
			v = np.zeros((x.size,y.size))
			for i in range(x.size):
				for j in range(y.size):
					v[i,j] = V(x[i], y[j])
		# Set values of ghost cells
		v[0, :]  = v[2, :]
		v[:, 0]  = v[:, 2]
		v[-1, :] = v[-3, :]
		v[:, -1] = v[:, -3]

		# Calculting the negative time step by a backward difference:
		up = u - dt*v

		# Allow f to be None or 0
		if f == None or f == 0:
			if version == 'scalar':
				f = lambda x, y, t: 0
			elif version == 'vectorized':
				f = lambda x, y, t: zeros((x.size, y.size))

		f = f if f else (lambda x, y, t: 0)

		# Make q into a numpy array
		q_array = zeros((x.size, y.size))
		if type(q) == float or type(q) == int:
			q_array.fill(q)
		else:
			for i in xrange(x.size):
				for j in xrange(y.size):
					q_array[i,j] = q(x[i],y[j])
			# Set q-values of ghost cells, assuming dq/dn=0 at boundary
			q_array[0,:] = q_array[2,:]
			q_array[:,0] = q_array[:,2]
			q_array[-1,:] = q_array[-3,:]
			q_array[:,-1] = q_array[:,-3]

		# Select version
		if version == "vectorized":
			self.advance = self.advance_vectorized

		# Store values for later use
		self.x, self.y, self.t = x, y, t
		self.Ix, self.Iy = Ix, Iy
		self.dx, self.dy, self.dt = dx, dy, dt
		self.u, self.up = u, up
		self.T = T
		self.hx, self.hy = hx, hy
		self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
		self.b = float(b)
		self.q = q_array
		self.f = f
		self.n = 0
		self.Lx, self.Ly = Lx, Ly


	def advance(self):
		up, upp = self.u, self.up
		x, y, t = self.x, self.y, self.t
		Ix, Iy = self.Ix,self.Iy
		hx, hy = self.hx, self.hy
		Nx, Ny, Nt = self.Nx, self.Ny, self.Nt
		dx, dy, dt = self.dx, self.dy, self.dt
		b, q, f = self.b, self.q, self.f
		n = self.n + 1
		u = np.zeros((Nx+3, Ny+3))

		# Updating the internal mesh points
		for i in xrange(1, self.Nx+2):
			for j in xrange(1, self.Ny+2):
				u[i,j] = 2./(2+b*dt)*( 2*up[i,j] - (1-b*dt/2.)*upp[i,j] \
					+ 0.5*hx*((q[i+1,j]+q[i,j])*(up[i+1,j]-up[i,j])		\
						+ (q[i-1,j]+q[i,j])*(up[i-1,j]-up[i,j]))		\
					+ 0.5*hx*((q[i,j+1]+q[i,j])*(up[i,j+1]-up[i,j]) 	\
						+ (q[i,j-1]+q[i,j])*(up[i,j-1]-up[i,j]))		\
					+ dt**2 * f(x[i],y[j],t[n]))
				
				# Setting the ghost values for the next step: 
				u[0,j]  = u[2,j]  
				u[i,0]  = u[i,2]  
				u[-1,j] = u[-3,j] 
				u[i,-1] = u[i,-3] 
				
		self.up = up
		self.u = u
		self.n = n

	def advance_vectorized(self):
		up, upp = self.u, self.up
		x, y, t = self.x, self.y, self.t
		hx, hy = self.hx, self.hy
		dt = self.dt
		b, q, f = self.b, self.q, self.f
		n = self.n + 1
		Nx, Ny = self.Nx, self.Ny
		u = np.zeros((Nx+3,Ny+3))

		u[1:-1,1:-1] = 2./(2+b*dt)*(2*up[1:-1,1:-1]-(1-b*dt/2.)*upp[1:-1,1:-1] \
			+ 0.5*hx*((q[2:,1:-1] + q[1:-1,1:-1])*(up[2:,1:-1]-up[1:-1,1:-1])
				+ (q[0:-2,1:-1] + q[1:-1,1:-1])*(up[0:-2,1:-1]-up[1:-1,1:-1]))
			+ 0.5*hy*((q[1:-1,2:]+q[1:-1,1:-1])*(up[1:-1,2:]-up[1:-1,1:-1]) 
				+ (q[1:-1,0:-2]+q[1:-1,1:-1])*(up[1:-1,0:-2]-up[1:-1,1:-1]))
			+ dt**2 * f(x[1:-1],y[1:-1],t[n]))

		# Setting the ghost values for the next step: 
		u[0, 1:-1]  = u[2, 1:-1]
		u[1:-1, 0]  = u[1:-1, 2]
		u[-1, 1:-1] = u[-3, 1:-1]
		u[1:-1, -1] = u[1:-1, -3]
	
		self.up = up
		self.u = u
		self.n = n

	def solve(self, plot=False):
		T, dt = self.T, self.dt
		n = int(ceil(T/dt))
		for i in range(n):
			self.advance()

	def get_mesh(self):
		return self.x[1:-1], self.y[1:-1]

	def get_solution(self):
		return self.u[1:-1,1:-1].T


class Plotter():
	def __init__(self, solver):
		self.solver = solver

	def plot_u(self, show=False, save=False):
		from mpl_toolkits.mplot3d import axes3d
		import matplotlib.pyplot as plt
		
		solver = self.solver
		X, Y = np.meshgrid(*solver.get_mesh())
		while solver.n < solver.Nt:
			print solver.n
			solver.advance()
			u = solver.get_solution()
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.plot_wireframe(X, Y, u, rstride=1, cstride=1)
			ax.set_xlabel(r'$x$', fontsize=22)
			ax.set_ylabel(r'$y$', fontsize=22)
			ax.set_zlabel(r'$u$', fontsize=22)
			ax.set_xlim(0, solver.Lx)
			ax.set_ylim(0, solver.Ly)
			#ax.set_zlim(0, 2)
			if save:
				plt.savefig("tmp/%s%04d.png" % (save, solver.n))
				self.figname = save
			if show:
				plt.show()
			plt.close()



def test_constant(C=pi):
	"""
	Test the solver using constant solution, u_e=C. No source term
	is then needed. Implements a nose test.
	"""
	import nose.tools as nt

	def I(x, y):
		return C

	V = 0.
	q = 2.
	Lx = 5
	Ly = 10
	Nx = 40
	Ny = 80
	dt = 0.1
	T = 10
	b = 3.0
	f = 0
	version = 'vectorized'

	for v in ["scalar", "vectorized"]:
		solver = Solver(I,V,q,Lx,Ly,Nx,Ny,dt,T,b=b,f=f,version=version)
		u_e = solver.get_solution()
		solver.solve()
		u = solver.get_solution()
		diff = abs(u-u_e).max()
		print diff
		nt.assert_almost_equal(diff, 0, places=10)

def test_gaussian():
	V = 0
	q = 20.0
	Lx = 5
	Ly = 10
	I = lambda x, y: 2*exp(-(x-0.5*Lx)**2-(y-0.5*Ly)**2/2)
	Nx = 40
	Ny = 80
	dt = 0.01
	T = 1
	b = 2.0
	version="vectorized"
	
	solver = Solver(I, V, q, Lx, Ly, Nx, Ny, dt, T, b=b, f=None, version=version)
	plotter = Plotter(solver)
	plotter.plot_u(save="gauss")

test_constant()

