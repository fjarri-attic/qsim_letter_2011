import numpy
import time
import math

from beclab import *
from beclab.state import ParticleStatistics, Uncertainty
from helpers import *


class AveragesCollector:

	def __init__(self, env, constants):
		self._env = env
		self._constants = constants
		self._stats = ParticleStatistics(env, constants)
		self._reduce = createReduce(env, constants.scalar.dtype)
		self._creduce = createReduce(env, constants.complex.dtype)

		self.times = []
		self.n1 = []
		self.n2 = []
		self.i = []

	def __call__(self, t, cloud):
		self.times.append(t)

		ensembles = cloud.a.size / self._constants.cells
		get = self._env.fromDevice
		reduce = self._reduce
		creduce = self._creduce
		dV = self._constants.dV

		i = self._stats._getInteraction(cloud.a, cloud.b)
		n1 = self._stats.getDensity(cloud.a)
		n2 = self._stats.getDensity(cloud.b)

		self.i.append(get(creduce(i, ensembles)) * dV)
		self.n1.append(get(reduce(n1, ensembles)) * dV)
		self.n2.append(get(reduce(n2, ensembles)) * dV)

	def getData(self):
		return numpy.array(self.times), self.i, self.n1, self.n2


def testUncertainties(a12, gamma12, losses):

	# normal parameters
	if losses:
		m = Model(N=44000, detuning=-37,
			nvx=8, nvy=8, nvz=64, dt_evo=4e-5, ensembles=1024, e_cut=1e6,
			fx=97.0, fy=97.0 * 1.03, fz=11.69,
			gamma12=gamma12, gamma22=7.7e-20,
			a12=a12, a22=95.57)
	else:
		m = Model(N=44000, detuning=-37,
			nvx=8, nvy=8, nvz=64, dt_evo=4e-5, ensembles=1024, e_cut=1e6,
			fx=97.0, fy=97.0 * 1.03, fz=11.69,
			gamma111=0, gamma12=0, gamma22=0,
			a12=a12, a22=95.57)

	t = 0.2
	callback_dt = 0.002
	noise = losses

	# Yun Li, shallow trap, no losses
	#m = Model(N=20000, nvx=16, nvy=16, nvz=16, ensembles=4, e_cut=1e6,
	#	a11=100.44, a12=88.28, a22=95.47, fx=42.6, fy=42.6, fz=42.6,
	#	gamma111=0, gamma12=0, gamma22=0
	#	)
	#t = 0.5
	#callback_dt = 0.001
	#noise = False

	# Yun Li, steep trap, no losses
	#m = Model(N=100000, nvx=16, nvy=16, nvz=16, ensembles=1024, e_cut=1e6,
	#	a11=100.44, a12=88.28, a22=95.47, fx=2e3, fy=2e3, fz=2e3, dt_evo=1e-6, dt_steady=1e-7,
	#	gamma111=0, gamma12=0, gamma22=0
	#	)
	#t = 0.005
	#callback_dt = 0.00005
	#noise = False

	#m = Model(N=100000, nvx=16, nvy=16, nvz=16, ensembles=512, e_cut=1e6,
	#	a11=100.0, a12=50.0, a22=100.0, fx=2e3, fy=2e3, fz=2e3, dt_evo=1e-6, dt_steady=1e-7,
	#	gamma111=0, gamma12=0, gamma22=0
	#	)
	#t = 0.005
	#callback_dt = 0.00005
	#noise = False

	constants = Constants(m, double=True)
	env = envs.cuda()
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)
	a = VisibilityCollector(env, constants, verbose=True)
	u = UncertaintyCollector(env, constants)
	b = AnalyticNoiseCollector(env, constants)
	c = AveragesCollector(env, constants)

	gs = GPEGroundState(env, constants)

	cloud = gs.createCloud()
	cloud.toWigner()

	pulse.apply(cloud, 0.5 * math.pi, matrix=True)

	t1 = time.time()
	evolution.run(cloud, t, callbacks=[a, u, b, c], callback_dt=callback_dt, noise=noise)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"


	#times, na, nb, sp = u.getData()
	#xi_sq = XYData("squeezing", times, numpy.log10(sp), xname="Time, s", yname="log$_{10}$($\\xi^2$)")
	#xi_sq.save('squeezing_ramsey.json')
	#XYPlot([xi_sq]).save("squeezing_ramsey.pdf")

	#times, _, overlap = b.getData()
	#ov = XYData("overlap", times, overlap, xname="T, s", yname="Overlap")
	#ov.save('squeezing_overlap.json')
	#XYPlot([ov]).save('squeezing_overlap.pdf')

	times, i, n1, n2 = c.getData()

	env.release()

	return times, i, n1, n2


def combinedTest(N, a12, gamma12, losses):
	t = None
	ii = None
	nn1 = None
	nn2 = None

	for i in xrange(N):
		times, i, n1, n2 = testUncertainties(a12, gamma12, losses)

		if ii is None:
			ii = i
			nn1 = n1
			nn2 = n2
			t = times
		else:
			for j in xrange(len(ii)):
				ii[j] = numpy.concatenate([ii[j], i[j]])
				nn1[j] = numpy.concatenate([nn1[j], n1[j]])
				nn2[j] = numpy.concatenate([nn2[j], n2[j]])

	xi = []

	if losses:
		m = Model(N=44000, detuning=-37,
			nvx=8, nvy=8, nvz=64, dt_evo=4e-5, ensembles=1024 * N, e_cut=1e6,
			fx=97.0, fy=97.0 * 1.03, fz=11.69,
			gamma12=gamma12, gamma22=7.7e-20,
			a12=a12, a22=95.4)
	else:
		m = Model(N=44000, detuning=-37,
			nvx=8, nvy=8, nvz=64, dt_evo=4e-5, ensembles=1024 * N, e_cut=1e6,
			fx=97.0, fy=97.0 * 1.03, fz=11.69,
			gamma111=0, gamma12=0, gamma22=0,
			a12=a12, a22=95.4)

	constants = Constants(m, double=True)
	env = envs.cuda()
	u = Uncertainty(env, constants)

	for j in xrange(len(ii)):
		xi.append(u._getXiSquared(ii[j], nn1[j], nn2[j]))

	env.release()

	return XYData("a12 = " + str(a12) + ("" if losses else ", no losses"),
		times * 1e3, 10 * numpy.log10(numpy.array(xi)),
		xname="T (ms)", yname="$\\xi^2$ (dB)")

xi1nl = combinedTest(1, 80.0, 7.3e-19, False)
xi2nl = combinedTest(1, 85.0, 3.67e-19, False)
xi3nl = combinedTest(1, 90.0, 1.34e-19, False)
xi4nl = combinedTest(1, 96.0, 1.53e-20, False)

xi1nl.color = 'red'
xi2nl.color = 'blue'
xi3nl.color = 'green'
xi4nl.color = 'black'

xi1nl.save('squeezing_ramsey_80.0_nolosses.json')
xi2nl.save('squeezing_ramsey_85.0_nolosses.json')
xi3nl.save('squeezing_ramsey_90.0_nolosses.json')
xi4nl.save('squeezing_ramsey_96.0_nolosses.json')

for x in (xi1nl, xi2nl, xi3nl, xi4nl):
	x.linestyle = '--'

xi1 = combinedTest(1, 80.0, 7.3e-19, True)
xi2 = combinedTest(1, 85.0, 3.67e-19, True)
xi3 = combinedTest(1, 90.0, 1.34e-19, True)
xi4 = combinedTest(1, 96.0, 1.53e-20, True)

xi1.color = 'red'
xi2.color = 'blue'
xi3.color = 'green'
xi4.color = 'black'

xi1.save('squeezing_ramsey_80.0.json')
xi2.save('squeezing_ramsey_85.0.json')
xi3.save('squeezing_ramsey_90.0.json')
xi4.save('squeezing_ramsey_96.0.json')

XYPlot([xi1nl, xi2nl, xi3nl, xi4nl, xi1, xi2, xi3, xi4],
	location="upper left").save('squeezing.pdf')
