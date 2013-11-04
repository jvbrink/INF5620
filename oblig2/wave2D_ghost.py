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
		u = self.update_ghost_cells(u, Ix, Iy)
		v = self.update_ghost_cells(v, Ix, Iy)
		
		# Make q into npdarray
		q = np.zeros((Nx+3,Ny+3))
		for i in Ix:
			for j in Iy:
				q[i,j] = problem.q(x[i],y[j])
		# Set q-values of ghost cells, assuming dq/dn=0 at boundary
		q = self.update_ghost_cells(q, Ix, Iy)
		
		# Select version
		if version == "vectorized":
			self.advance = self.advance_vectorized

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
		self.n = 0

		# Perform first step
		self.first_step()

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

				# print 2*uij
				# print (1 - b*dt/2.)*2*dt*v[i,j]
				# print dt2*f(x[i], y[j], t[n])
				# print hx*u_x + hy*u_y

				u[i,j] = 2*uij + (1 - b*dt/2.)*2*dt*v[i,j]  \
				               + hx*u_x + hy*u_y + dt2*f(x[i],y[j],t[n])

				u[i,j] /= 2.
				
		u = self.update_ghost_cells(u, Ix, Iy)
		
		self.up = up
		self.u = u
		self.n = n + 1 

	def advance(self):
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

		# print 2*uij # ulik
		# print (b*dt/2. - 1) * upp[1:-1,1:-1]  # lik
		# print dt2*f(x[1:-1],y[1:-1],t[n]) # lik
		# print hx*u_x + hy*u_y # ulik

		u[1:-1,1:-1] = 2*uij + (b*dt/2. - 1)*upp[1:-1,1:-1] \
		               + hx*u_x + hy*u_y + dt2*f(x[1:-1],y[1:-1],t[n])

		u[1:-1,1:-1] /= (1 + b*dt/2.)

		# Updating ghost cells
		u[0, :] = u[2, :]; u[-1,:] = u[-3,:]
		u[:, 0] = u[:, 2]; u[:,-1] = u[:,-3]

		self.up = up
		self.u = u
		self.n = n + 1

	def solve(self, plot=False):
		T, dt = self.T, self.dt
		n = int(np.ceil(T/dt))
		for i in range(n):
			self.advance()

	def get_mesh(self):
		return self.x[1:-1], self.y[1:-1]

	def get_solution(self):
		return self.u[1:-1,1:-1]

def plot_solution(x, y, u):
	from mpl_toolkits.mplot3d import axes3d
	import matplotlib.pyplot as plt

	X, Y = np.meshgrid(x, y)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(X, Y, u.T, rstride=1, cstride=1)
	ax.set_xlabel(r'$x$', fontsize=22)
	ax.set_ylabel(r'$y$', fontsize=22)
	ax.set_zlabel(r'$u$', fontsize=22)
	#ax.set_xlim(0, solver.Lx)
	#ax.set_ylim(0, solver.Ly)
	plt.show()
	plt.close()

def plot_solutions(solver, show=False, save=False, zlim=None):
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
		ax.plot_wireframe(X, Y, u.T, rstride=1, cstride=1)
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

def plot_error(solver, u_e, show=False, save=False, zlim=None):
	from mpl_toolkits.mplot3d import axes3d
	import matplotlib.pyplot as plt
	
	solver = solver
	X, Y = np.meshgrid(*solver.get_mesh())
	u = solver.get_solution()
	e = abs(u-u_e)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(X, Y, e.T, rstride=1, cstride=1)
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
