"""
2D wave equation solver
"""


from pylab import *


class Solver():
	"""Solver class for wave equation"""

	def __init__(self, I):

		x = linspace()
		y = linspace()

		# Evaluate initial condition in meshpoints
		up = I(x,y)






	def advance():
		up, upp = self.u, self.up
		q = self.q
		hx, hy = self.hx, self.hy

		u[i,j] = 2*up[i,j] - upp[i,j] + \
	 	hx*(q(i+0.5,j)*(up[i+1,j]-up[i,j]) - q(i-0.5,j)*(up[i,j]-up[i-1,j])) + \
	 	hy*(q(i,j+0.5)*(up[i,j+1]-up[i,j]) - q(i,j-0.5)*(up[i,j]-up[i,j-1])) + \
	 		f(i, j, n)
		
		self.up = up
		self.u = u
		

