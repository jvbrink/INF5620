from numpy import *

class Problem:
	def __init__(self, params = {
		'rho' 	: 1.3163,	# density of fluid (kg/m^3)
		'C_D' 	: 0.45,		# drag coeff of body (dim. less)
		'm'   	: 1.0,      # mass of body (kg)
		'A'   	: pi,		# cross-sectional area perp to motion (m^2
		'g'   	: 9.81},	# acceleration of gravity (m/s/s)	
		**new_params): 

		params.update(new_params)
		self.params = params

	def exact_solution(self, t):
		raise NotImplementedError

class Solver:
	def __init__(self, problem, dt=0.001, T=10.0, v0=0.0):
		self.problem = problem
		dt = float(dt)

		C_D = problem.params['C_D']
		rho = problem.params['rho']
		A = problem.params['A']
		m = problem.params['m']
		g = problem.params['g']

		# Constants used in numerical schemes
		a = C_D*rho*A/(2*m)
		b = g
		self.prob_constants = [a,b]

		self.k = 0				     # step number
		self.dt = dt 				 # time step (s)
		self.T = T 					 # end time (s)
		self.N = int(ceil(T/dt)) + 1 # number of mesh points
		self.v = zeros(self.N) 	     # velocity array
		self.v[0] = v0 			     # initial velocity
		self.t = linspace(0, T, self.N)	 # time array

	def step(self):
		v = self.v
		k = self.k
		dt = self.dt
		a, b = self.prob_constants

		# Step using numerical scheme
		v[k+1] = (v[k] + b*dt)/(1 + a*dt*abs(v[k]))

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
	problem = Problem(A=1., m=80.)
	solver = Solver(problem, dt=0.001, T=20.0)
	v, t = solver.solve()

	from pylab import *
	plot(t, v)
	xlabel(r'$t$ (s)', fontsize=16)
	ylabel(r'$v$ (m/s)', fontsize=16)
	title(r'Fallrate of a skydiver', fontsize=16)
	axis([t[0], t[-1], v[0], 1.1*v[-1]])
	show()