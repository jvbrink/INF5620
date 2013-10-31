"""
INF5620 Mandatory Assignment 1
Exercise 24
Making a program for vertical motion 
in a fluid and verify it using nose
"""

from numpy import *


class Problem:
	def __init__(self, params = {
		'rho' 	: 1000.,	# density of fluid (kg/m^3)
		'mu'  	: 1.002e-3, # viscosity of fluid (N s/m2)
		'C_D' 	: 0.45,		# drag coeff of body (dim. less)
		'd'	  	: 0.1,		# diameter of spherical body (m)
		'm'   	: 1.0,      # mass of body (kg)
		'V'   	: 4./3*pi,	# volume of body (m^3)
		'A'   	: pi,		# cross-sectional area perp to motion (m^2
		'g'   	: 9.81},	# acceleration of gravity (m/s/s)	
		**new_params): 

		params.update(new_params)
		self.params = params

	def exact_solution(self, t):
		raise NotImplementedError


class Solver:
	def __init__(self, problem, dt=0.001, T=0.5, v0=0.0):
		self.problem = problem
		
		rho = problem.params['rho']
		mu = problem.params['mu']
		C_D = problem.params['C_D']
		m = problem.params['m']
		d = problem.params['d']
		V = problem.params['V']
		A = problem.params['A']
		g = problem.params['g']

		# Constants used in numerical schemes
		a = 3*pi*d*mu / m
		b = g*(rho*V / m - 1)
		c = 0.5*C_D*(rho*A)/m
		self.prob_constants = [a,b,c]

		self.k = 0				     # step number
		dt = float(dt)
		self.dt = dt 				 # time step (s)
		self.T = T 					 # end time (s)
		self.N = int(ceil(T/dt)) + 1 # number of mesh points
		self.Re = zeros(self.N) 	 # array for Reynolds number
		self.v = zeros(self.N) 	     # velocity array
		self.v[0] = v0 			     # initial velocity
		self.Re[0] = self.reynolds_number() # initial Re
		self.t = linspace(0, T, self.N)	 # time array

	def reynolds_number(self):
		# Find Reynold's number for current velocity
		rho = problem.params['rho']
		mu = problem.params['mu']
		d = problem.params['d']
		m = problem.params['m']
		g = problem.params['g']
		
		k = self.k				
		v = self.v[k]			

		return rho*d*abs(v)/mu

	def step(self):
		v = self.v
		k = self.k
		dt = self.dt
		a, b, c = self.prob_constants

		if self.Re[k] < 1.:
			# Step using Stokes' drag
			v[k+1] = ((1-a*dt/2.)*v[k] + b*dt)/(1 + a*dt/2.)
			print "Stokes"
		
		else:
			# Step using quadratic drag
			v[k+1] = (v[k] + b*dt)/(1 + c*dt*abs(v[k]))

		# Update values
		self.k += 1
		self.v = v

		# Find Re for new velocity
		self.Re[k+1] = self.reynolds_number()


	def solve(self):
		# Solve problem for t in (0, T]
		N = self.N

		for i in range(N-1):
			self.step()

		return self.v, self.t

# Example of use and testing
if __name__ == '__main__':
	r = 0.11 	
	d = 2*r
	V = 4./3*pi*r**3
	A = pi*r*r
	m = 0.43	
	
	problem = Problem(d=d, V=V, A=A, m=m)
	solver = Solver(problem, dt=0.001, T=0.1)
	v, t = solver.solve()

	from pylab import *
	plot(t, v)
	xlabel(r'$t$ (s)', fontsize=16)
	ylabel(r'$v$ (m/s)', fontsize=16)
	title(r'Vertical motion through a fluid', fontsize=16)
	axis([t[0], t[-1], v[0], 1.1*v[-1]])
	show()

	plot(t, solver.Re)
	xlabel(r'$t$ (s)', fontsize=16)
	ylabel(r'$Re$', fontsize=16)
	title(r'Vertical motion through a fluid', fontsize=16)
	show()