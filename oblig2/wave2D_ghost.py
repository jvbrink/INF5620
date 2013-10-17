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

	def __init__(self, I, V, q, Lx, Ly, Nx, Ny, dt, T, b=0.0, f=None):
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

		# Make sure source term is callable
		f = f if f else (lambda x, y, t: 0)

		# Make sure q is a numpy array of correct size
		q_array = zeros((x.size, y.size))
		if type(q) == float or type(q) == int:
			q_array[:,:] += q
		else:
			for i in xrange(x.size):
				for j in xrange(y.size):
					q_array[i,j] = q(x[i],y[j])
			# Set q-values of ghost cells, assuming dq/dn=0 at boundary
			q_array[0,:] = q_array[2,:]
			q_array[:,0] = q_array[:,2]
			q_array[-1,:] = q_array[-3,:]
			q_array[:,-1] = q_array[:,-3]


		# calculting the negative time step by a backward difference:
		up = u-dt*v 

		# Store values for later use
		self.x = x
		self.y = y
		self.t = t
		self.Ix = Ix
		self.Iy = Iy
		self.dx = dx
		self.dy = dy
		self.dt = dt
		self.u = u
		self.up = up
		self.hx = hx
		self.hy = hy
		self.Nx = Nx
		self.Ny = Ny
		self.Nt = Nt
		self.b = b
		self.q = q_array
		self.f = f
		self.n = 0


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
		#for i in Ix:  # i goes from 1 to Nx	
		for i in xrange(1, self.Nx+2):
			for j in xrange(1, self.Ny+2):
			#for j in Iy: # j goes from 1 to Ny
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

	def solve(self):
		n = int(ceil(T/dt))
		for i in range(n):
			advance()


V = 0
q = 20.0
Lx = 5
Ly = 10
I = lambda x, y: 2*exp(-(x-0.5*Lx)**2-(y-0.5*Ly)**2/2)
Nx = 40
Ny = 80
dt = 0.01
T = 2
test = Solver(I, V, q, Lx, Ly, Nx, Ny, dt, T, b=0., f=None)

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


X, Y = np.meshgrid(test.x[1:-1], test.y[1:-1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, test.u[1:-1,1:-1].T, rstride=1, cstride=1)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_zlim(0, 2)
plt.show()
#plt.savefig("tmp/fig%04d.png" % test.n)

print test.x[1], test.y[1]


while test.n < test.Nt:
	test.advance()
	print test.n
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(X, Y, test.u[1:-1,1:-1].T, rstride=1, cstride=1)
	ax.set_xlim(0, Lx)
	ax.set_ylim(0, Ly)
	ax.set_zlim(0,2)
	plt.savefig("tmp/fig%04d.png" % test.n)


"""
plt.pcolor(test.x, test.y, test.u.T)
plt.axis([0,Lx,0,Ly])
plt.colorbar()
plt.savefig("tmp/fig%s.png" % str(test.n))

while test.n < test.Nt:
	test.advance()
	plt.pcolor(test.x, test.y, test.u.T)
	plt.axis([0,Lx,0,Ly])
	plt.savefig("tmp/fig%s.png" % str(test.n))
	print test.n
"""

#os.system("mencoder 'mf://*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o test.mpg")

