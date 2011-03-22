import numpy
import time
import math

from beclab import *
from beclab.state import ParticleStatistics, Uncertainty

def runPass(evolution, pulse, cloud, t):
	pulse.apply(cloud, math.pi, matrix=True)
	evolution.run(cloud, t, noise=False)

def testRephasing(wigner):

	if wigner:
		nvx = nvy = 8
		nvz = 64
		dt_evo = 4e-5
	else:
		nvx = nvy = 16
		nvz = 128
		dt_evo = 1e-5

	m = Model(N=44000, detuning=-37,
		nvx=nvx, nvy=nvy, nvz=nvz, dt_evo=dt_evo, ensembles=1024, e_cut=1e6,
		fx=97.0, fy=97.0 * 1.03, fz=11.69,
		gamma12=1.52e-20, gamma22=7.7e-20,
		a12=97.93, a22=95.4)

	t_max = 2.0
	t_step = 0.05

	constants = Constants(m, double=True)
	env = envs.cuda(device_num=1)
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)

	stats = ParticleStatistics(env, constants)
	unc = Uncertainty(env, constants)

	gs = GPEGroundState(env, constants)

	times = [0.0]
	vis = [1.0]
	xi_sq = []

	cloud = gs.createCloud()
	if wigner:
		cloud.toWigner()
		xi_sq.append(unc.getXiSquared(cloud.a, cloud.b))

	pulse.apply(cloud, 0.5 * math.pi, matrix=True)

	name = 'squeezing_rephasing'

	t = t_step
	while t <= t_max:
		evolution.run(cloud, t_step / 2, noise=wigner)
		print "Pre-pi step finished, t=" + str(cloud.time)
		new_cloud = cloud.copy()
		pulse.apply(new_cloud, math.pi, matrix=True)
		evolution.run(new_cloud, t / 2, noise=wigner)
		print "Post-pi step finished"

		times.append(t)
		vis.append(stats.getVisibility(new_cloud.a, new_cloud.b))
		print "Visibility=" + str(vis[-1])

		if wigner:
			xi_sq.append(unc.getXiSquared(new_cloud.a, new_cloud.b))

		XYData(name, numpy.array(times), numpy.array(vis),
			xname="T, s", yname="Visibility").save(name + '_vis.json')

		if wigner:
			XYData(name, numpy.array(times), numpy.log10(numpy.array(xi_sq)), ymin=0,
				xname="T, s", yname="log$_{10}(\\xi^2)$").save(name + '_xi_sq.json')

		XYPlot([XYData.load(name + '.json')]).save(name + '.pdf')

		del new_cloud
		t += t_step

	env.release()


testRephasing(True)
