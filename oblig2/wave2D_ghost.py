#!/usr/bin/env python

import numpy as np
from scitools.std import *

class Solver():
	"""
	A solver for solving a standard 2D linear wave eguation
	with damping:

	u_tt + b*u_t = (q(x,y)*u_x)_x + (q(x,y)u_y)_y + f(x,y,t)

	with boundary condition: 	u_n = 0

	and  initial conditions: 	u(x,y,0) = I(x,y)
								u_t(x,y,0) = V(x,y)
	"""

	def __init__(self, I, V, q, Lx, Ly, Nx, Ny, dt, T, f=None):
		# Create the meshpoints with ghost points
		#x = np.linspace(0, Lx, Nx+1) # x-dir
		#y = np.linspace(0, Ly, Ny+1) # y-dir

		x = np.linspace(0, Lx, Nx+3) # x-dir
		y = np.linspace(0, Ly, Ny+3) # y-dir
		Ix = range(1,x.size-1)
		Iy = range(1,y.size-1)
		dx = x[1] - x[0]
		dy = y[1] - y[0]
		Nt = int(round(T/float(dt)))
		t = np.linspace(0, Nt*dt, Nt+1) # time
		hx = (dt/dx)**2
		hy = (dt/dx)**2
		
    	# Set up initial conditions
		if type(I)==np.ndarray: # what about the size?
			u = I
		else:
			# Make sure I is callable
			I = I if I else (lambda x, y: 0)
			# Evaluate I in mesh points
			u = np.zeros((x.size,y.size))
			for i in range(x.size):
				for j in range(y.size):
					u[i,j] = I(x[i],y[j])
		

		if type(V) == np.ndarray:
			v = V
		else:
			# Make sure V is callable
			V = V if V else (lambda x, y: 0)
			# Evaluate V in mesh points
			v = np.zeros((x.size,y.size))
			for i in range(x.size):
				for j in range(y.size):
					v[i,j] = V(x[i],y[j])

		# Make sure f and q are callable		
		if type(q) == float or type(q) == int:
			tmp = q
			q = (lambda x, y: tmp)
		f = f if f else (lambda x, y, t: 0)


		# calculting the negative time step by a centered difference:
		up = u-2*dt*v 

		# Store values for later use
		self.x = x
		self.y = y
		self.Ix = Ix
		self.Iy = Iy
		self.dx = dx
		self.dy = dy
		self.u = u
		self.up = up
		self.hx = hx
		self.hy = hy
		self.Nx = Nx
		self.Ny = Ny
		self.Nt = Nt
		self.q = q
		self.f = f
		self.n = 0


	def advance(self):
		up, upp = self.u, self.up
		Ix, Iy = self.Ix,self.Iy
		hx, hy = self.hx, self.hy
		Nx, Ny, Nt = self.Nx, self.Ny, self.Nt
		dx, dy = self.dx, self.dy
		q, f = self.q, self.f
		n = self.n + 1
		u = np.zeros((Nx+3, Ny+3))


		# Updating the internal mesh points
		for i in Ix:  # i goes from 1 to Nx		
			for j in Iy: # j goes from 1 to Ny
				u[i,j] = 2*up[i,j] - upp[i,j] + \
						hx*(q((i+.5)*dx,j*dy)*(up[i+1,j] - up[i,j]) - \
							q((i-.5)*dx,j*dy)*(up[i,j] - up[i-1,j])) + \
						hy*(q(i*dx,(j+.5)*dy)*(up[i,j+1] - up[i,j]) - \
							q(i*dx,(j-.5)*dy)*(up[i,j] - up[i,j-1])) + \
						f(i*dx,j*dy,n*Nt)
				# Setting the ghost values for the next step: 
				u[0,j] = u[2,j] # u(-1,j)=u(1,j)
				u[Nx+2,j] = u[Nx,j] # u(Nx+1,j) = u(Nx-1,j)
			u[i,0] = u[i,2] # u(i,-1)=u(i,1)
			u[i,Ny+2] = u[i,Ny] # u(i,Ny+1)=u(i,Ny-1)
				
				

		'''
		# Setting the ghost values for the next step: 
		u[0,:] = u[2,:] # u(-1,j)=u(1,j)
		u[Nx+2,:] = u[Nx,:] # u(Nx+1,j) = u(Nx-1,j)
		u[:,0] = u[:,2] # u(i,-1)=u(i,1)
		u[:,Ny+2] = u[:,Ny] # u(i,Ny+1)=u(i,Ny-1)
		'''


		self.up = up
		self.u = u
		self.n = n

	def solve(self):
		n = int(ceil(T/dt))
		for i in range(n):
			advance()



I = lambda x, y: sin(x) + cos(y)
V = 0
q = 0
Lx = 5
Ly = 5
Nx = 40
Ny = 40
dt = 0.1
T = 2
test = Solver(I, V, q, Lx, Ly, Nx, Ny, dt, T, f=None)

import matplotlib.pyplot as plt
import os

plt.pcolor(test.u.T)
plt.colorbar()

while test.n < 5:
	test.advance()
	plt.pcolor(test.u.T)
	plt.savefig("tmp/fig%s.png" % str(test.n))
	print test.n

#os.system("mencoder 'mf://*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o test.mpg")

