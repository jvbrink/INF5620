#!/usr/bin/env python

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

class WaveProblem():
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

class WaveSolver():
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
		u = self.update_ghost_cells(u, Ix, Iy)
		v = self.update_ghost_cells(v, Ix, Iy)
		
		# Make q into npdarray
		q = np.zeros((Nx+3,Ny+3))
		for i in Ix:
			for j in Iy:
				q[i,j] = problem.q(x[i],y[j])
		# Set q-values of ghost cells, assuming dq/dn=0 at boundary
		q = self.update_ghost_cells(q, Ix, Iy)

		# First step uses a different scheme
		self.advance = self.first_step

		# Store values for later use
		self.x, self.y, self.t = x, y, t
		self.Ix, self.Iy = Ix, Iy
		self.dx, self.dy, self.dt = dx, dy, dt
		self.u = u
		self.v = v
		self.T = T
		self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
		self.Lx, self.Ly = Lx, Ly
		self.b = problem.b
		self.q = q
		self.f = problem.f
		self.version = version
		self.n = 0

	

	def update_ghost_cells(self, u, Ix, Iy):
		i = Ix[0]
		u[i-1,:] = u[i+1,:]
		i = Ix[-1]
		u[i+1,:] = u[i-1,:]
		j = Iy[0]
		u[:,j-1] = u[:,j+1]
		j = Iy[-1]
		u[:,j+1] = u[:,j-1]

		return u
		

	def first_step(self):
		up = self.u
		x, y, t = self.x, self.y, self.t
		Ix, Iy = self.Ix, self.Iy
		hx, hy = self.dt**2/self.dx**2, self.dt**2/self.dy**2
		b, q, f = self.b, self.q, self.f
		dt = self.dt
		v = self.v
		dt2 = dt**2
		
		n = self.n
		u = np.zeros(up.shape)
		for i in Ix:
			for j in Iy:
				# Updating the internal mesh points
				qij = q[i,j]
				qpi = (q[i+1,j] + qij)/2.
				qmi = (q[i-1,j] + qij)/2.
				qpj = (q[i,j+1] + qij)/2.
				qmj = (q[i,j-1] + qij)/2.
				uij = up[i,j]

				u_x = qpi*(up[i+1,j] - uij) + qmi*(up[i-1,j] - uij)
				u_y = qpj*(up[i,j+1] - uij) + qmj*(up[i,j-1] - uij)

				u[i,j] = 2*uij + (1 - b*dt/2.)*2*dt*v[i,j]  \
				         + hx*u_x + hy*u_y + dt2*f(x[i],y[j],t[n])

				u[i,j] /= 2.
				
		u = self.update_ghost_cells(u, Ix, Iy)

		self.up = up
		self.u = u
		self.n = n + 1 

		# Select version
		if self.version == "scalar":
			self.advance = self.advance_scalar
		if self.version == "vectorized":
			self.advance = self.advance_vectorized

	def advance(self):
		raise NotImplementedError

	def advance_scalar(self):
		up, upp = self.u, self.up
		x, y, t = self.x, self.y, self.t
		Ix, Iy = self.Ix, self.Iy
		hx, hy = self.dt**2/self.dx**2, self.dt**2/self.dy**2
		b, q, f = self.b, self.q, self.f
		dt = self.dt
		dt2 = dt**2
		
		n = self.n
		u = np.zeros(up.shape)
		for i in Ix:
			for j in Iy:
				# Updating the internal mesh points
				qij = q[i,j]
				qpi = (q[i+1,j] + qij)/2.
				qmi = (q[i-1,j] + qij)/2.
				qpj = (q[i,j+1] + qij)/2.
				qmj = (q[i,j-1] + qij)/2.
				uij = up[i,j]

				u_x = qpi*(up[i+1,j] - uij) + qmi*(up[i-1,j] - uij)
				u_y = qpj*(up[i,j+1] - uij) + qmj*(up[i,j-1] - uij)


				u[i,j] = 2*uij + (b*dt/2. - 1)*upp[i,j] \
				               + hx*u_x + hy*u_y + dt2*f(x[i],y[j],t[n])

				u[i,j] /= (1 + b*dt/2.)

				
		u = self.update_ghost_cells(u, Ix, Iy)
		
		self.up = up
		self.u = u
		self.n = n+1

	def advance_vectorized(self):
		up, upp = self.u, self.up
		x, y, t = self.x, self.y, self.t
		hx, hy = self.dt**2/self.dx**2, self.dt**2/self.dy**2
		b, q, f = self.b, self.q, self.f
		dt = self.dt
		dt2 = dt**2
		n = self.n
		u = np.zeros(up.shape)

		# Updating the internal mesh points
		qij = q[1:-1,1:-1]
		qpi = (q[2:,1:-1] + qij)/2.
		qmi = (q[:-2,1:-1] + qij)/2.
		qpj = (q[1:-1,2:] + qij)/2.
		qmj = (q[1:-1,:-2] + qij)/2.
		uij = up[1:-1,1:-1]

		u_x = qpi*(up[2:,1:-1] - uij) + qmi*(up[:-2,1:-1] - uij)
		u_y = qpj*(up[1:-1,2:] - uij) + qmj*(up[1:-1,:-2] - uij)

		u[1:-1,1:-1] = 2*uij + (b*dt/2. - 1)*upp[1:-1,1:-1] \
		               + hx*u_x + hy*u_y + dt2*f(x[1:-1],y[1:-1],t[n])

		u[1:-1,1:-1] /= (1 + b*dt/2.)

		u = self.update_ghost_cells(u, self.Ix, self.Iy)

		self.up = up
		self.u = u
		self.n = n + 1

	def solve(self):
		T, dt = self.T, self.dt
		n = int(np.ceil(T/dt))
		for i in range(n):
			self.advance()

	def get_mesh(self):
		return self.x[1:-1], self.y[1:-1]

	def get_solution(self):
		return self.u[1:-1,1:-1]


class WavePlotter:
	def __init__(self, solver):
		self.solver = solver
		self.X, self.Y = np.meshgrid(*solver.get_mesh())

	def plot_solution(self, show=True, save=False, stride=[1,1], zlim=None):
		X, Y, solver = self.X, self.Y, self.solver
		u = solver.get_solution()

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_wireframe(X, Y, u.T, rstride=stride[0], cstride=stride[1])
		ax.set_xlabel(r'$x$', fontsize=22)
		ax.set_ylabel(r'$y$', fontsize=22)
		ax.set_zlabel(r'$u$', fontsize=22)
		ax.set_xlim(0, solver.Lx)
		ax.set_ylim(0, solver.Ly)
		if zlim:
			ax.set_zlim(0, zlim)
		if show:
			plt.show()
		if save:
			plt.savefig("tmp/%s%04d.png" % (save, solver.n))
		plt.close()

	def solve_and_plot(self, stride=[1,1], show=False, save=False, zlim=None):
		X, Y, solver = self.X, self.Y, self.solver

		print "Solving and plotting problem."
		while solver.n < solver.Nt:
			self.plot_solution(show=show, save=save, zlim=zlim, stride=stride)
			self.solver.advance()
			print "Finished with step %d of %d." % (solver.n, solver.Nt)
