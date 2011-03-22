import numpy
import time
import math

from beclab import *
from beclab.state import ParticleStatistics

def testUncertainties():

	# normal parameters
	m = Model(N=44000, detuning=-37,
		nvx=8, nvy=8, nvz=64, dt_evo=4e-5, ensembles=1024, e_cut=1e6,
		fx=97.0, fy=97.0 * 1.03, fz=11.69,
		gamma12=1.52e-20, gamma22=7.7e-20,
		a12=97.93, a22=95.4)

	t = 2.0
	callback_dt = 0.002
	noise = True

	constants = Constants(m, double=True)
	env = envs.cuda()
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)
	a = VisibilityCollector(env, constants, verbose=True)
	u = UncertaintyCollector(env, constants)

	gs = GPEGroundState(env, constants)

	cloud = gs.createCloud()
	cloud.toWigner()

	pulse.apply(cloud, 0.5 * math.pi, matrix=True)

	t1 = time.time()
	evolution.run(cloud, t, callbacks=[a, u], callback_dt=callback_dt, noise=noise)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	times, na, nb, sp = u.getData()

	xi_sq = XYData("squeezing", times, numpy.log10(sp), xname="Time, s", yname="log$_{10}$($\\xi^2$)")
	xi_sq.save('squeezing_ramsey.json')

	XYPlot([xi_sq]).save("squeezing_ramsey.pdf")

	env.release()

testUncertainties()
