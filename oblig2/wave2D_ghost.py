#!/usr/bin/env python

import numpy as np
#from scitools.std import *

class Problem():
	def __init__(self, b, q_const=1.0):
		self.b = b
		self.q_const = q_const

	def f(self, x, y, t):
		"""Source term"""
		return 0.0

	def I(self, x, y):
		"""Initial condition, u(x,y,0)=I(x,y)"""
		return 0.0

	def V(self, x, y):
		"""Initial velocity u_t(x,y,0)=V(x,y)"""
		return 0.0

	def q(self, x, y):
		"""Wave velocity, q=c^2"""
		return self.q_const

class Solver():
	"""
	A solver for solving a standard 2D linear wave equation
	with damping:

	u_tt + b*u_t = (q(x,y)*u_x)_x + (q(x,y)u_y)_y + f(x,y,t)

	with boundary condition: 	u_n = 0

	and  initial conditions: 	u(x,y,0) = I(x,y)
								u_t(x,y,0) = V(x,y)
	"""

	def __init__(self, problem, Lx, Ly, Nx, Ny, dt, T, version="scalar"):
		# Create spatial mesh including ghost points
		dx = Lx/float(Nx)
		dy = Ly/float(Ny)
		x = np.linspace(-dx, Lx+dx, Nx+3)
		y = np.linspace(-dy, Ly+dy, Ny+3)
		Ix = range(1, Nx+2)
		Iy = range(1, Ny+2)

		# Create time mesh
		Nt = int(round(T/float(dt)))
		t = np.linspace(0, Nt*dt, Nt+1)

		# Set up inital conditions
		u = np.zeros((Nx+3, Ny+3))
		v = np.zeros((Nx+3, Ny+3))
		for i in Ix:
			for j in Iy:
				u[i,j] = problem.I(x[i],y[j])
				v[i,j] = problem.V(x[i],y[j])

		# Set values of ghost cells
		u[0, :] = u[2, :]; u[-1,:] = u[-3,:]
		u[:, 0] = u[:, 2]; u[:,-1] = u[:,-3]
		v[0, :] = v[2, :]; v[:, 0] = v[:, 2]
		v[-1,:] = v[-3,:]; v[:,-1] = v[:,-3]

		# Calculting the negative time step by a backward difference:
		up = u - dt*v

		# Make q into npdarray
		q = np.zeros((Nx+3,Ny+3))
		for i in Ix:
			for j in Iy:
				q[i,j] = problem.q(x[i],y[j])
		# Set q-values of ghost cells, assuming dq/dn=0 at boundary
		q[0, :] = q[2, :]; q[:, 0] = q[:,2]
		q[-1,:] = q[-3,:]; q[:,-1] = q[:,-3]

		# Select version
		if version == "vectorized":
			self.advance = self.advance_vectorized

		# Store values for later use
		self.x, self.y, self.t = x, y, t
		self.Ix, self.Iy = Ix, Iy
		self.dx, self.dy, self.dt = dx, dy, dt
		self.u, self.up = u, up
		self.T = T
		self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
		self.Lx, self.Ly = Lx, Ly
		self.b = problem.b
		self.q = q
		self.f = problem.f
		self.n = 0

	def advance(self):
		up, upp = self.u, self.up
		x, y, t = self.x, self.y, self.t
		Ix, Iy = self.Ix, self.Iy
		hx, hy = self.dt**2/self.dx**2, self.dt**2/self.dy**2
		b, q, f = self.b, self.q, self.f
		dt = self.dt
		
		n = self.n+1
		u = np.zeros(up.shape)
		for i in Ix:
			for j in Iy:
				# Updating the internal mesh points
				u[i,j] = 2./(2+b*dt)*(2*up[i,j] - (1-b*dt/2.)*upp[i,j] 	\
					+ 0.5*hx*((q[i+1,j]+q[i,j])*(up[i+1,j]-up[i,j])	   	\
						+ (q[i-1,j]+q[i,j])*(up[i-1,j]-up[i,j]))		\
					+ 0.5*hx*((q[i,j+1]+q[i,j])*(up[i,j+1]-up[i,j]) 	\
						+ (q[i,j-1]+q[i,j])*(up[i,j-1]-up[i,j]))		\
					+ dt**2*f(x[i],y[j],t[n]))

				# Updating ghost values
				u[0, j] = u[2, j]; u[i, 0] = u[i, 2]  
				u[-1,j] = u[-3,j]; u[i,-1] = u[i,-3] 

		self.up = up
		self.u = u
		self.n = n

	def advance_vectorized(self):
		up, upp = self.u, self.up
		x, y, t = self.x, self.y, self.t
		hx, hy = self.dt**2/self.dx**2, self.dt**2/self.dy**2
		dt = self.dt
		b, q, f = self.b, self.q, self.f
		
		n = self.n + 1
		u = np.zeros(up.shape)

		# Updating the internal mesh points
		u[1:-1,1:-1] = 2./(2+b*dt)*(2*up[1:-1,1:-1]-(1-b*dt/2.)*upp[1:-1,1:-1] \
			+ 0.5*hx*((q[2:,1:-1] + q[1:-1,1:-1])*(up[2:,1:-1]-up[1:-1,1:-1])
				+ (q[0:-2,1:-1] + q[1:-1,1:-1])*(up[0:-2,1:-1]-up[1:-1,1:-1]))
			+ 0.5*hy*((q[1:-1,2:]+q[1:-1,1:-1])*(up[1:-1,2:]-up[1:-1,1:-1]) 
				+ (q[1:-1,0:-2]+q[1:-1,1:-1])*(up[1:-1,0:-2]-up[1:-1,1:-1]))
			+ dt**2 * f(x[1:-1],y[1:-1],t[n]))

		# Updating ghost cells
		u[0, 1:-1] = u[2, 1:-1]; u[1:-1, 0] = u[1:-1, 2]
		u[-1,1:-1] = u[-3,1:-1]; u[1:-1,-1] = u[1:-1,-3]
	
		self.up = up
		self.u = u
		self.n = n

	def solve(self, plot=False):
		T, dt = self.T, self.dt
		n = int(np.ceil(T/dt))
		for i in range(n):
			self.advance()

	def get_mesh(self):
		return self.x[1:-1], self.y[1:-1]

	def get_solution(self):
		return self.u[1:-1,1:-1].T

def plotSolutions(solver, show=False, save=False, zlim=None):
	from mpl_toolkits.mplot3d import axes3d
	import matplotlib.pyplot as plt
	
	solver = solver
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
		if zlim:
			ax.set_zlim(0, zlim)
		if save:
			plt.savefig("tmp/%s%04d.png" % (save, solver.n))
			figname = save
		if show:
			plt.show()
		plt.close()


def gaussianPeak():
	class GaussianWithDamping(Problem):
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
	T = 1
	b = 2.0
	version="vectorized"
	problem = GaussianWithDamping(Lx, Ly, b)
	solver = Solver(problem, Lx, Ly, Nx, Ny, dt, T, version=version)
	plotSolutions(solver, save='test', zlim=2)
