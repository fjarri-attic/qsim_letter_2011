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

	m = Model(N=55000, detuning=-37,
		nvx=nvx, nvy=nvy, nvz=nvz, dt_evo=dt_evo, ensembles=64, e_cut=1e6,
		fx=97.0, fy=97.0 * 1.03, fz=11.69,
		gamma12=1.53e-20, gamma22=7.7e-20,
		a12=97.99, a22=95.57)

	t_max = 8.0
	t_step = 0.05

	constants = Constants(m, double=True)
	env = envs.cuda(device_num=0)
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)

	stats = ParticleStatistics(env, constants)
	unc = Uncertainty(env, constants)

	gs = GPEGroundState(env, constants)

	times = [0.0]
	vis = [1.0]
	phnoise = []
	phis = []
	ypss = []

	cloud = gs.createCloud()
	if wigner:
		cloud.toWigner()

		phnoise.append(stats.getPhaseNoise(cloud.a, cloud.b))
		phi, yps = unc.getSpins(cloud.a, cloud.b)
		phis.append(phi)
		ypss.append(yps)

	pulse.apply(cloud, 0.5 * math.pi, matrix=True)

	if wigner:
		name = 'rephasing_wigner'
	else:
		name = 'rephasing_gpe'

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
			phnoise.append(stats.getPhaseNoise(new_cloud.a, new_cloud.b))
			phi, yps = unc.getSpins(new_cloud.a, new_cloud.b)
			phis.append(phi)
			ypss.append(yps)

		XYData(name, numpy.array(times), numpy.array(vis),
			ymin=0, ymax=1, xname="T, s", yname="$\\mathcal{V}$").save(name + '_vis.json')

		if wigner:
			XYData(name, numpy.array(times), numpy.array(phnoise), ymin=0,
				xname="T, s", yname="Phase noise, rad").save(name + '_phnoise.json')

			Data('spins',
				['name', 'phi', 'yps', 'time', 'timename', 'phiname', 'ypsname'],
				time=numpy.array(times), phi=numpy.array(phis), yps=numpy.array(ypss), timename="T (s)",
				phiname="Spin, azimuth, rad", ypsname="Spin, inclination, rad",
				name=name).save(name + "_phnoise_points.pickle")

		XYPlot([XYData.load(name + '_vis.json')]).save(name + '_vis.pdf')

		del new_cloud
		t += t_step

	env.release()


testRephasing(False)
testRephasing(True)

vis_gpe = XYData.load('rephasing_gpe_vis.json')
vis = XYData.load('rephasing_wigner_vis.json')
XYPlot([vis_gpe, vis]).save('visibility.pdf')
