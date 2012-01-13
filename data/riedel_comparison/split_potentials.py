"""
This example demonstrates usage of static per-component potentials
"""

import numpy
from beclab import *
from beclab.constants import buildProjectorMask

# parameters from Riedel at al. (2010)
N = 1250
f_rad = 109
f_ax = 500
f_rabi = 2100
f_detuning = -40
potentials_separation = 0.52e-6
splitting_time = 12.7e-3
e_cut = 5000
shape = (128, 16, 16)
parameters = repr(dict(e_cut=e_cut, shape=shape))


def split_potentials(constants, grid):

	x = grid.x_full
	y = grid.y_full
	z = grid.z_full

	potentials = lambda dz: constants.m * (
		(constants.wx * x) ** 2 +
		(constants.wy * y) ** 2 +
		(constants.wz * (z + dz)) ** 2) / (2.0 * constants.hbar)

	return numpy.concatenate([
		potentials(potentials_separation / 2),
		potentials(-potentials_separation / 2)
	]).reshape((2,) + x.shape).astype(constants.scalar.dtype)


def runPass():
	env = envs.cuda(device_num=1)
	constants = Constants(double=env.supportsDouble(),
		a11=100.4, a12=97.7, a22=95.0, fx=f_ax, fy=f_ax, fz=f_rad, e_cut=e_cut)

	# axial size of the N / 2 cloud ~ 8e-6 >> potentials_separation
	# therefore we can safely use normal grid, provided that it has big enough border
	box = constants.boxSizeForN(N, 3, border=1.2)
	grid = UniformGrid(env, constants, shape,
		(box[0] + potentials_separation * 2, box[1], box[2]))
	#print constants.planeWaveModesForCutoff(constants.boxSizeForN(N, 3, border=2))

	gs = SplitStepGroundState(env, constants, grid, dt=1e-6)
	pulse = Pulse(env, constants, grid, f_rabi=f_rabi, f_detuning=f_detuning)
	evolution = SplitStepEvolution(env, constants, grid,
		potentials=split_potentials(constants, grid),
		dt=1e-6)

	n = ParticleNumberCollector(env, constants, grid, verbose=False)
	v = VisibilityCollector(env, constants, grid)
#	a = AxialProjectionCollector(env, constants, grid, pulse=pulse)
	u = UncertaintyCollector(env, constants, grid)
	s = SpinCloudCollector(env, constants, grid)

	psi = gs.create((N, 0))

	psi.toMSpace()
	mode_data = numpy.abs(env.fromDevice(psi.data))[0, 0] # remember mode data
	mask = buildProjectorMask(constants, grid)
	psi.toXSpace()

	psi.toWigner(128)
	pulse.apply(psi, numpy.pi / 2)
	evolution.run(psi, splitting_time, callbacks=[v, n, u, s],
		callback_dt=splitting_time / 100)
	env.synchronize()
	env.release()

	"""
	times, heightmap = a.getData()
	HeightmapPlot(
		HeightmapData("test", heightmap,
			xmin=0, xmax=splitting_time * 1e3,
			ymin=grid.z[0] * 1e6,
			ymax=grid.z[-1] * 1e6,
			zmin=-1, zmax=1,
			xname="T (ms)", yname="z ($\\mu$m)", zname="Spin projection")
	).save('split_potentials_axial.pdf')
	"""

	times, n_stddev, xi_squared = u.getData()
	XYData("Squeezing", times * 1000, numpy.log10(xi_squared),
		xname="T (ms)", yname="log$_{10}$($\\xi^2$)",
		description=parameters).save('split_potentials_xi.json')


	times, vis = v.getData()
	XYData('test', times * 1e3, vis,
		xname="T (ms)", yname="$\\mathcal{V}$",
		ymin=0, ymax=1,
		description=parameters).save('split_potentials_vis.json')


	times, phi, yps, Sx, Sy, Sz = s.getData()
	return times, Sx, Sy, Sz


if __name__ == '__main__':

	Sxs = []
	Sys = []
	Szs = []
	times = None

	for i in xrange(100):
		print "*** Running ", i + 1, "-th pass"
		times, Sx, Sy, Sz = runPass()
		XYPlot([XYData.load('split_potentials_xi.json')]).save('split_potentials_xi_squared.pdf')
		XYPlot([XYData.load('split_potentials_vis.json')]).save('split_potentials_vis.pdf')
		Sxs.append(Sx)
		Sys.append(Sy)
		Szs.append(Sz)

		Sx = numpy.concatenate(Sxs, 1)
		Sy = numpy.concatenate(Sys, 1)
		Sz = numpy.concatenate(Szs, 1)

		Data('spins', ['times', 'name', 'Sx', 'Sy', 'Sz', 'description'],
			times=times, Sx=Sx, Sy=Sy, Sz=Sz,
			name="Spins", description="parameters").save('split_potentials_spins.pickle')
