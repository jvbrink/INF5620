from numpy import *

class Problem:
	def __init__(self, v0=0.0, m=90., C_D=1.4, rho=1.0, A=0.5, g=9.81):
		self.v0 = v0
		self.m = m
		self.C_D = C_D
		self.rho = rho
		self.A = A
		self.g = g

	def define_command_line_options(self, parser=None):
		if parser is None:
			import argparse
			parser = argparse.ArgumentParser()

		parser.add_argument(
			'--v0', '--initial_velocity', type=float,
			default=self.v0, help='initial velocity, v(0)',
			metavar='v0')

		parser.add_argument(
			'--m', '--mass', type=float,
			default=self.m, help='mass of body',
			metavar='m')
        
		parser.add_argument(
			'--C_D', '--drag_coeff', type=float,
			default=self.C_D, help='drag coefficient of body',
			metavar='C_D')

		parser.add_argument(
			'--rho', '--density', type=float,
			default=self.rho, help='density of fluid',
			metavar='rho')
        
		parser.add_argument(
			'--A', '--cross_section', type=float,
			default=self.A, help='cross-sectional area of body perp to motion',
			metavar='A')

		parser.add_argument(
			'--g', '--gravity', type=float,
			default=self.g, help='acceleration of gravity',
			metavar='g')
        
		return parser

	def init_from_command_line(self, args):
		self.v0, self.m, self.C_D, self.rho, self.A, self.g = args.v0, args.m, args.C_D, args.rho, args.A, args.g

	def exact_solution(self, t):
		b, c = self.b, self.v0
		return b*t + c	
       
	def source_term(self, t, b=2.0):
		self.b = b
		c = self.v0
		a = self.C_D * self.rho * self.A / 2.
		m, g = self.m, self.g

		return a * abs(b*t+c) * (b*t+c) + m * (b - g)

class Solver:
	def __init__(self, problem, dt=10e-4, T=10.0):
		
		N = ceil(T/float(dt)) + 1
		v = zeros(N)
		t = linspace(0, T, N)

		self.problem = problem
		self.k = 0				     # step number
		self.dt = dt 				 # time step (s)
		self.T = T 					 # end time (s)
		self.N = int(ceil(T/dt)) + 1 # number of mesh points
		self.v = zeros(self.N) 	     # velocity array
		self.v[0] = problem.v0 	     # initial velocity (m/s)
		self.t = linspace(0, T, self.N)	 # time array
		self.f = problem.source_term
	
	def define_command_line_options(self, parser):
		parser.add_argument(
			'--dt', '--time_step', type=float,
			default=self.dt, help='time step',
			metavar='dt')
        
		parser.add_argument(
			'--T', '--end_time', type=float,
			default=self.T, help='end time',
			metavar='T')

		return parser

	def init_from_command_line(self, args):
		self.dt, self.T = args.dt, args.T
		

	def step(self):
		v = self.v
		k = self.k
		f = self.f
		dt = self.dt

		prob = self.problem
		a = prob.C_D * prob.rho * prob.A / 2.
		m, g = prob.m, prob.g

		v[k+1] = (v[k] + dt * (g + f((k+0.5)*dt)/m)) / (1. + a*dt*abs(v[k])/m)

		self.k += 1
		self.v = v

	def solve(self):
		# Solve problem for t in (0, T]
		self.v[0] = self.problem.v0 # initial velocity (m/s)

		print self.dt, self.N-1
		for i in range(self.N-1):
			self.step()

		return self.v, self.t

class Visualizer:
	def __init__(self, problem, solver):
		import matplotlib.pyplot as plt
		self.plt = plt
		self.problem, self.solver = problem, solver

	def plot(self, include_exact=False):
		plt = self.plt

		plt.plot(solver.t, solver.v)
		plt.xlabel(r'$t$ (s)', fontsize=16)
		plt.ylabel(r'$v$ (m/s)', fontsize=16)
		plt.title(r'Fallrate of a skydiver', fontsize=16)
		#plt.axis([solver.t[0], solver.t[-1], 0, 1.1*max(solver.v)])
		plt.grid()
		plt.show()

	def plot_error(self):
		plt = self.plt

		error = abs(solver.v - problem.exact_solution(solver.t))
		print max(error)
		
		plt.plot(solver.t, log10(error))
		plt.xlabel(r'$t$ (s)', fontsize=16)
		plt.ylabel(r'$\log_{10}(u-u_e)$', fontsize=16)
		plt.title(r'Error using MMS', fontsize=16)
		plt.legend([r'b=%g, c=%g' % (problem.b, problem.v0)], 4)
		plt.grid()
		plt.show()

	def plot_exact(self):
		plt = self.plt

		plt.plot(solver.t, problem.exact_solution(solver.t))
		plt.show()

# Example of use and testing
if __name__ == '__main__':
	problem = Problem()

	Ep = 1
	dtp = 1
	for dt in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
		solver = Solver(problem, dt=dt)
		v, t = solver.solve()

		e = (v - problem.exact_solution(solver.t))**2
		E = sqrt(sum(e*dt))

		r = (log(E) - log(Ep))/(log(dt) - log(dtp))

		Ep = E
		dtp = dt

		print "dt = %g         E = %g      r = %g" % (dt, E, r)


	'''
	parser = problem.define_command_line_options()
	parser = solver.define_command_line_options(parser)
	args = parser.parse_args()
	problem.init_from_command_line(args)
	solver.init_from_command_line(args)
	
	vis = Visualizer(problem, solver)
	solver.solve()
	vis.plot()
	vis.plot_error()
	'''

	