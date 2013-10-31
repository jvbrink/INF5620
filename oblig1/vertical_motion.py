"""
INF5620 Mandatory Assignment 1
Exercise 24
Making a program for vertical motion 
in a fluid and verify it using nose
"""

from numpy import *


class VerticalFluidMotion:
	"""Class implementing both
	the problem and solver"""

	def __init__(self, dt=0.01, T=10., v0=0., rho=1000., mu=1.002e-3, C_D=0.45, d=0.1, m=1., V=4./3*pi, A=pi, g=9.81):
		# rho - density of fluid (kg/m^3)
		# mu  - viscosity (N s/m2)
		# C_D - Drag coefficient (unitless)
		# b   - diameter of body perp to flow (m)
		# m   - mass of body (kg)
		# V   - volume of body (m^3)
		# g   - acceleration of gravity (m/s/s)	
		self.params = [rho, mu, d, m, V, g]

		# Constants used in numerical schemes
		a = 3*pi*d*mu / m
		b = g*(rho*V / m - 1)
		c = 0.5*C_D*(rho*A)/m
		self.prob_constants = [a,b,c]

		self.k = 0				# Step number
		self.dt = dt 			# Time step, s
		self.T = T 				# End time, s
		self.N = int(ceil(T/dt))# Number of cells
		self.Re = zeros(self.N) # Array for Reynolds number
		self.v = zeros(self.N) 	# Velocity array
		self.v[0] = v0 			# Initial velocity
		self.t = arange(0, T, dt)	# Time array


	def reynolds_number(self):
		# Find Reynold's number for current velocity
		rho, mu, d, m, V, g = self.params
		k = self.k
		v = self.v[k]
		return rho*d*abs(v)/mu

	def step(self):
		v = self.v
		k = self.k
		dt = self.dt
		a, b, c = self.prob_constants

		# Find Re
		self.Re[k] = self.reynolds_number()

		if self.Re[k] < 1.:
			# Step using Stokes' drag
			v[k+1] = ((1-a*dt/2.)*v[k] + b*dt)/(1 + a*dt/2.)
		
		else:
			# Step using quadratic drag
			v[k+1] = (v[k] + b*dt)/(1 + c*dt*abs(v[k]))

		# Update values
		self.k += 1
		self.v = v


	def solve(self):
		# Solve problem for t in (0, T]
		N = self.N

		for i in range(N-1):
			self.step()

		return self.v, self.t

# Example of use and testing
if __name__ == '__main__':
	r = 0.11 	# Radius, meter
	d = 2*r
	V = 4./3*pi*r**3
	m = 0.43	# Mass of soccer ball, kg
	rho = 1000 # Density of water, kg/m^3
	mu = 0.45 # drag coefficient, unitless

	problem = VerticalFluidMotion(dt=0.001, T=0.1, m=m, rho=rho, d=d, V=V, A=pi*r*r)
	v, t = problem.solve()

	from pylab import *
	plot(t, v)
	show()