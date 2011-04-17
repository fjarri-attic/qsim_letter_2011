import numpy
import time
import math

from beclab import *

def calculateRamsey(wigner, echo):

	t = 8.0

	if wigner:
		nvx = nvy = 8
		nvz = 64
		dt_evo = 4e-5
	else:
		nvx = nvy = 16
		nvz = 128
		dt_evo = 1e-5

	m = Model(N=55000, detuning=-37,
		nvx=nvx, nvy=nvy, nvz=nvz, dt_evo=dt_evo, ensembles=256, e_cut=1e6,
		fx=97.0, fy=97.0 * 1.03, fz=11.69,
		gamma12=1.53e-20, gamma22=7.7e-20,
		a12=97.99, a22=95.57)

	constants = Constants(m, double=True)
	env = envs.cuda(device_num=0)
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)

	a = VisibilityCollector(env, constants, verbose=True)
	b = PhaseNoiseCollector(env, constants)
	c = SpinCloudCollector(env, constants)

	collectors = [a, b, c] if wigner else [a]

	gs = GPEGroundState(env, constants)

	cloud = gs.createCloud()
	if wigner:
		cloud.toWigner()

	pulse.apply(cloud, 0.5 * math.pi, matrix=True)

	t1 = time.time()
	if echo:
		evolution.run(cloud, t / 2, callbacks=collectors, callback_dt=0.002, noise=wigner)
		pulse.apply(cloud, math.pi, matrix=True)
		evolution.run(cloud, t / 2, callbacks=collectors, callback_dt=0.002, noise=wigner)
	else:
		evolution.run(cloud, t, callbacks=collectors, callback_dt=0.002, noise=wigner)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	name = ("Spin-echo sequence" if echo else "Ramsey sequence") + ", " + \
		("Wigner" if wigner else "GPE")

	times, vis = a.getData()
	vis_data = XYData(name, times, vis, xname="T, s", ymin=0, ymax=1, yname="$\\mathcal{V}$")

	if wigner:
		times, noise = b.getData()
		noise_data = XYData(name, times, noise, xname="T (s)", ymin=0,
			yname="Phase noise, rad")

		times, phi, yps = c.getData()
		noise_points = Data('spins', ['phi', 'yps', 'time', 'timename', 'phiname', 'ypsname'],
			time=time, phi=phi, yps=yps, timename="T, s",
			phiname="Spin, azimuth, rad", ypsname="Spin, inclination, rad")

	env.release()

	if wigner:
		return vis_data, noise_data, noise_points
	else:
		return vis_data

v1 = calculateRamsey(False, False)
v1.save('ramsey_gpe_vis.json')

v3, n3, n3_points = calculateRamsey(True, False)
v3.save('ramsey_wigner_vis.json')
n3.save('ramsey_wigner_phnoise.json')
n3_points.save('ramsey_wigner_phnoise_points.pickle')

XYPlot([v1, v3]).save('visibility.pdf')
