"""
2D wave equation solver
"""

import numpy as np

class Solver():
	"""
	Class for solving the 2D wave equation with
	damping and a spatially variable wave velocity

	u_tt + b*u_t = (q(x,y)*u_x)_x + (q(x,y)*u_y)_y + f(x,y,t)

	Constructor:
	I 		- Initial u
	V 		- Initial velocity
	f 	 	- Source term
	q		- Wave velocity, q=c^2
	Lx, Ly 	- Lengths
	Nx, Ny 	- Number of mesh cells
	dt 		- Time step
	T 		- End time
	"""

	def __init__(self, I, V, f, q, Lx, Ly, Nx, Ny, dt, T):

		# Create meshpoints
		x = np.linspace(0, Lx, Nx+1)
		y = np.linspace(0, Ly, Ny+1)
		
		# Calculate spacing
		dx = x[1]-x[0]
		dy = y[1]-y[0]
		hx = (dt/dx)**2
		hy = (dt/dx)**2
		Nt = int(round(T/float(dt)))
		t = np.linspace(0, Nt*dt, Nt+1)

		# Set up initial conditions
		if type(I) == np.ndarray:
			u = I
		else:
			# Make sure I is callable
			I = I if I else (lambda x, y: 0)
			# Evaluate I in mesh points
			u = zeros((Nx+1,Ny+1))
			for i in range(Nx+1):
				for j in range(Ny+1)
					u[i,j] = I(x[i],y[j])
			

		if type(V) == np.ndarray:
			v = V
		else:
			# Make sure V is callable
			V = V if V else (lambda x, y: 0)
			# Evaluate V in mesh points
			v = zeros((Nx+1,Ny+1))
			for i in range(Nx+1):
				for j in range(Ny+1)
					v[i,j] = V(x[i],y[j])
			
		# Calculate the ??? point
		up = u - dt*v

		# Store values for later use
		self.x = x
		self.y = y
		self.u = u
		self.up = up
		self.hx = hx
		self.hy = hy
		self.Nt = Nt


	def advance():
		up, upp = self.u, self.up
		hx, hy = self.hx, self.hy
		Nx, Ny = self.Nx, self.Ny
		dx, dy = self.dx, self.dy
		q = self.q

		# Update internal mesh points
		for i in range(1, Nx):
			for j in range(1, Ny):
				u[i,j] = 2*up[i,j] - upp[i,j] + \
			 		hx*(q((i+.5)*dx,j*dy)*(up[i+1,j]-up[i,j]) -\
			 			q((i-.5)*dx,j*dy)*(up[i,j]-up[i-1,j])) +\
			 		hy*(q(i*dx,(j+.5)*dy)*(up[i,j+1]-up[i,j]) -\
			 			q(i*dx,(j-.5)*dy)*(up[i,j]-up[i,j-1]))+\
		 			f(i*dx, j*dy, n*Nt)

	 	# Update boundary mesh points
		for j in range(1, Ny):
			# x=0 boundary
			u[0,j] = 2*up[0,j] - upp[0,j] + \
			 		2*hx*(q(.5*dx,j*dy)*(up[1,j]-up[0,j]) +\
			 		hy*(q(0,(j+.5)*dy)*(up[0,j+1]-up[0,j]) -\
			 			q(0,(j-.5)*dy)*(up[0,j]-up[0,j-1]))+\
		 			f(0, j*dy, n*Nt)

			# x=Lx boundary
		 	u[-1,j] = 2*up[-1,j] - upp[-1,j] + \
			 		2*hx*(q((Nx-.5)*dx,j*dy)*(up[-2,j]-up[-1,j]) +\
			 		hy*(q(0,(j+.5)*dy)*(up[-1,j+1]-up[-1,j]) -\
			 			q(0,(j-.5)*dy)*(up[-1,j]-up[-1,j-1]))+\
		 			f(Nx*dx, j*dy, n*Nt)

		for i in range(1, Nx):
			# y=0 boundary
			u[i,0] = 2*up[i,0] - upp[i,0] + \
			 		hx*(q((i+.5)*dx,0)*(up[i+1,0]-up[i,0]) -\
			 			q((i-.5)*dx,0)*(up[i,0]-up[i-1,0])) +\
			 		2*hy*(q(i*dx,.5*dy)*(up[i,1]-up[i,0]) +\
		 			f(i*dx, Nx*dx, n*Nt)

		 	# y=Ly boundary
		 	u[i,-1] = 2*up[i,-1] - upp[i,-1] + \
			 		hx*(q((i+.5)*dx,Ny*dy)*(up[i+1,-1]-up[i,-1]) -\
			 			q((i-.5)*dx,Ny*dy)*(up[i,-1]-up[i-1,-1])) +\
			 		2*hy*(q(i*dx,(Ny-.5)*dy)*(up[i,-2]-up[i,-1]) +\
		 			f(i*dx, Ny*dy, n*Nt)			

		# Corners
		u[0,0] = 2*up[0,0] - upp[0,0] + \
			 		2*hx*(q(.5*dx,0)*(up[1,0]-up[0,0]) +\
			 		2*hy*(q(0,.5*dy)*(up[0,1]-up[0,0]) +\
		 			f(0, 0, n*Nt)
		u[-1,0] = 2*up[-1,0] - upp[-1,0] + \
			 		2*hx*(q((Nx-.5)*dx,0)*(up[-2,0]-up[0,0]) +\
			 		2*hy*(q((Nx-.5)*dx,.5*dy)*(up[0,1]-up[0,0]) +\
		 			f(Nx*dx, 0, n*Nt)
		u[0,-1] = 2*up[0,-1] - upp[0,-1] + \
			 		2*hx*(q(.5*dx,Ny*dy)*(up[1,-1]-up[0,-1]) +\
			 		2*hy*(q(0,(Ny-.5)*dy)*(up[0,-2]-up[0,-1]) +\
		 			f(0, Ny*dy, n*Nt)
		u[-1,-1] = 2*up[-1,-1] - upp[-1,-1] + \
			 		2*hx*(q((Nx-.5)*dx,Ny*dy)*(up[-2,-1]-up[-1,-1]) +\
			 		2*hy*(q(Nx*dx,(Ny-.5)*dy)*(up[-1,-2]-up[-1,-1]) +\
		 			f(Nx*dx, Ny*dy, n*Nt)



		
		self.up = up
		self.u = u

	def solve():
		n = int(ceil(T/dt))
		for i in range(n):
			advance()