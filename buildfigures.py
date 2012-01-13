import numpy
import json
import math
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.path as path

from scipy.interpolate import interp1d
import copy

# Size and number of bins for spin cloud
cloud_xsize = 100.0
cloud_ysize = 250.0
cloud_zsize = 50.0
cloud_zbins = 25
cloud_levels = 20

cloud_ybins = int(float(cloud_zbins) * cloud_ysize / cloud_zsize + 0.5)
cloud_xbins = int(cloud_zbins * cloud_xsize / cloud_zsize + 0.5)
cloud_ybins = int(cloud_zbins * cloud_ysize / cloud_zsize + 0.5)


def updateParams(aspect=None):
	fig_width_pt = 0.9 * 246.0
	inches_per_pt = 1.0 / 72.27               # Convert pt to inches
	golden_mean = (math.sqrt(5) - 1.0) / 2.4         # Aesthetic ratio
	fig_width = fig_width_pt * inches_per_pt  # width in inches
	fig_height = fig_width * (golden_mean if aspect is None else aspect) # height in inches
	fig_size = [fig_width, fig_height]
	params = {
	    'backend': 'ps',
	    'axes.labelsize': 7,
	    'axes.linewidth': 0.35,
	    'font.family': 'serif',
	    'text.fontsize': 7,
	    'legend.fontsize': 7,
	    'xtick.labelsize': 6,
	    'ytick.labelsize': 6,
	    'text.usetex': False,
	    'figure.figsize': fig_size,
	}
	matplotlib.rcParams.update(params)

def combineNoises(data1, xarray, yarray):
	data2_interp = interp1d(numpy.array(xarray), numpy.array(yarray),
		kind="cubic", bounds_error=False)
	data = copy.deepcopy(data1)

	data['yarray'] = numpy.sqrt(numpy.array(data['yarray']) ** 2 +
		data2_interp(numpy.array(data['xarray'])) ** 2).tolist()

	return data

def plotXYGraph(datasets, linetypes, name, xmin=None, xmax=None, ymin=None, ymax=None):

	fig = plt.figure()

	a = 0.19
	b = 0.23
	axes = [a, b, 0.95-a, 0.96-b]

	subplot = fig.add_axes(axes, xlabel=datasets[0]['xname'], ylabel=datasets[0]['yname'])

	subplot.set_xlim(xmin=xmin, xmax=xmax)
	subplot.set_ylim(ymin=ymin, ymax=ymax)

	for dataset, linetype in zip(datasets, linetypes):

		if 'yerrors' not in dataset:

			kwds = dict(label=dataset['name'])
			if linetype.endswith('--'):
				kwds['dashes'] = (6, 3) # default value is (6, 6)
			elif linetype.endswith('-.'):
				kwds['dashes'] = (5, 3, 1, 3) # default value is (3, 5, 1, 5)

			subplot.plot(
				numpy.array(dataset['xarray']),
				numpy.array(dataset['yarray']),
				linetype, **kwds)
		else:
			subplot.scatter(
				numpy.array(dataset['xarray']),
				numpy.array(dataset['yarray']),
				edgecolors="None", s=5, c=linetype[0]
			)
			subplot.errorbar(
				numpy.array(dataset['xarray']),
				numpy.array(dataset['yarray']),
				label=dataset['name'],
				yerr=numpy.array(dataset['yerrors']),
				linewidth=0.75, capsize=1.0,
				linestyle="None", c=linetype[0])

	fig.savefig(name)

def getHeightmap(X, Y, xmin, xmax, ymin, ymax, xbins, ybins):
	"""Returns heightmap, extent and levels for contour plot"""

	hist, xedges, yedges = numpy.histogram2d(X, Y, bins=(xbins, ybins),
		range=[[xmin, xmax], [ymin, ymax]])
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

	hmax = hist.max()
	levels = numpy.exp(numpy.arange(cloud_levels + 1) / float(cloud_levels) * numpy.log(hmax))
	return hist.T, extent, levels

def plotMainView(fname, Sx, Sy, Sz):
	"""Plot spin cloud and some supporting information"""

	updateParams(aspect=0.3)
	fig = plt.figure()

	a = 0.19
	b = 0.23
	axes = [a, b, 0.95-a, 0.96-b]
	subplot = fig.add_axes(axes, aspect=1)

	# optimal squeezing angle and corresponding variance, to plot supporting info
	min_angle = 9.6 / 180 * numpy.pi
	min_var = numpy.sqrt(25.349280804)

	# parameters for arrow pointing at the best squeezing
	arrow_len = 30
	arrow1_x = -(min_var + arrow_len) * numpy.sin(min_angle)
	arrow1_y = (min_var + arrow_len) * numpy.cos(min_angle)
	arrow1_dx = (arrow_len) * numpy.sin(min_angle)
	arrow1_dy = -(arrow_len) * numpy.cos(min_angle)

	arrow_kwds = dict(width=2.5, linewidth=0.3,
		shape="full",
		overhang=0, head_starts_at_zero=False, fill=False,
		length_includes_head=True,
		facecolor='blue')

	# supporting lines

	r = numpy.array([0, 1.0])

	# axis of the ellipse
	l1_x = (r * 2 - 1) * min_var * numpy.sin(min_angle)
	l1_y = (-r * 2 + 1) * min_var * numpy.cos(min_angle)

	# horizontal line
	l2_x = r * 150
	l2_y = r * 0

	# projection direction line
	l3_x = r * 150
	l3_y = r * 150 * numpy.sin(min_angle)

	# plot the cloud
	hm, extent, levels = getHeightmap(Sy, Sz, -cloud_ysize, cloud_ysize, -cloud_zsize, cloud_zsize, cloud_ybins, cloud_zbins)
	subplot.contourf(hm, extent=extent, cmap=cm.PuRd, levels=levels)

	# plot pointing arrows
	subplot.arrow(arrow1_x, arrow1_y, arrow1_dx, arrow1_dy, **arrow_kwds)
	subplot.arrow(-arrow1_x, -arrow1_y, -arrow1_dx, -arrow1_dy, **arrow_kwds)

	# plot supporting lines
	subplot.plot(l1_x, l1_y, color='blue', linewidth=0.3)
	subplot.plot(l2_x, l2_y, linestyle='--', color='black', linewidth=0.3, dashes=(6, 2))
	subplot.plot(l3_x, l3_y, linestyle='--', color='black', linewidth=0.3, dashes=(6, 2))

	# mark angle
	arc = matplotlib.patches.Arc((0.0, 0.0), 100, 100,
		theta1=0, theta2=min_angle / numpy.pi * 180, linewidth=0.3, fill=False)
	subplot.add_patch(arc)

	# plot labels
	subplot.text(-30, 20, "$d_\\theta$", fontsize=7)
	subplot.text(40, 10, "$\\theta$", fontsize=7)

	subplot.set_xlabel('$S_y$')
	subplot.set_ylabel('$S_z$')
	subplot.set_xlim(xmin=-cloud_ysize, xmax=cloud_ysize)
	subplot.set_ylim(ymin=-cloud_zsize, ymax=cloud_zsize)
	fig.savefig(fname)

def plot3DView(fname, Sx, Sy, Sz):
	"""Plot 3 views of the cloud"""

	nullfmt = NullFormatter()
	fig_width = 8

	x_d = 0.12
	x_dd = 0.03
	x_ly = (1.0 - x_d - x_dd * 2) / (1.0 + cloud_zsize / cloud_xsize)
	x_lx = (1.0 - x_d - x_dd * 2) / (1.0 + cloud_xsize / cloud_zsize)

	y_d = x_d
	y_dd = x_dd
	y_lx = x_lx
	y_lz = y_lx / cloud_xsize * cloud_zsize

	aspect = (x_d + x_dd * 2 + x_ly + x_lx) / (y_d + y_dd * 2 + y_lx + y_lz)
	y_d *= aspect
	y_dd *= aspect
	y_lx *= aspect
	y_lz *= aspect

	# definitions for the axes
	rectYZ = [x_d, y_d, x_ly, y_lz]
	rectXY = [x_d, y_d + y_dd + y_lz, x_ly, y_lx]
	rectXZ = [x_d + x_dd + x_ly, y_d, x_lx, y_lz]

	# start with a rectangular Figure
	fig = plt.figure(10, figsize=(fig_width, fig_width / aspect))

	axYZ = plt.axes(rectYZ)
	axXY = plt.axes(rectXY)
	axXZ = plt.axes(rectXZ)

	# no labels
	axXY.xaxis.set_major_formatter(nullfmt)
	axXZ.yaxis.set_major_formatter(nullfmt)

	hm, extent, levels = getHeightmap(Sy, Sz, -cloud_ysize, cloud_ysize, -cloud_zsize, cloud_zsize, cloud_ybins, cloud_zbins)
	axYZ.contourf(hm, extent=extent, cmap=cm.PuRd, levels=levels)
	axYZ.set_xlabel('$S_y$')
	axYZ.set_ylabel('$S_z$')
	axYZ.set_xlim(xmin=-cloud_ysize, xmax=cloud_ysize)
	axYZ.set_ylim(ymin=-cloud_zsize, ymax=cloud_zsize)

	hm, extent, levels = getHeightmap(Sy, Sx, -cloud_ysize, cloud_ysize, -cloud_xsize, cloud_xsize, cloud_ybins, cloud_xbins)
	axXY.contourf(hm, extent=extent, cmap=cm.PuRd, levels=levels)
	axXY.set_ylabel('$S_x$')
	axXY.set_xlim(xmin=-cloud_ysize, xmax=cloud_ysize)
	axXY.set_ylim(ymin=-cloud_xsize, ymax=cloud_xsize)

	hm, extent, levels = getHeightmap(Sx, Sz, -cloud_xsize, cloud_xsize, -cloud_zsize, cloud_zsize, cloud_xbins, cloud_zbins)
	axXZ.contourf(hm, extent=extent, cmap=cm.PuRd, levels=levels)
	axXZ.set_xlabel('$S_x$')
	axXZ.set_xlim(xmin=-cloud_xsize, xmax=cloud_xsize)
	axXZ.set_ylim(ymin=-cloud_zsize, ymax=cloud_zsize)

	fig.savefig(fname)

def buildRiedelTomographyPath():
	"""Process outlined blue line from Fig.3 in Riedel 2010"""
	xstart = 137.35189
	ystart = 90.287253

	xn90 = 137.35189
	x360 = 992.539
	y15 = 109.64235
	y0 = 398.97217

	yscale = (y15 - y0) / 15
	xscale = (x360 - xn90) / (360 + 90)

	points = "0,0 14.50878,-1.568829 24.01256,-1.804477 9.50378,-0.235648 34.72699,2.276605 64.63501,37.401344 29.90803,35.12473 57.92996,97.02575 73.67738,186.27925 15.74742,89.2535 26.89408,328.94804 26.89408,328.94804 0,0 11.84222,-239.32902 27.17539,-324.70817 15.33317,-85.37915 37.30822,-145.80606 66.7308,-182.33072 29.42259,-36.524652 53.98301,-45.175188 75.48187,-45.648008 21.49886,-0.47282 42.77044,3.737291 73.61912,38.420108 30.84868,34.68281 51.95238,95.04074 69.15619,181.47674 17.20381,86.436 28.81508,331.37343 28.81508,331.37343 0,0 11.86912,-254.44324 30.73609,-338.09694 18.86697,-83.6537 33.03585,-128.95471 60.85126,-163.90638 27.81541,-34.95167 47.80063,-48.936916 79.15907,-49.548283 31.35845,-0.611367 50.95538,13.159523 79.60519,47.287693 28.64981,34.12817 53.63199,120.14375 61.81177,163.28546 8.17978,43.14171 12.82612,84.24291 12.82612,84.24291"
	points = points.split(' ')
	points = [tuple([float(x) for x in point.split(',')]) for point in points]

	vertices = [[xstart, ystart]]
	codes = [path.Path.MOVETO]
	while len(points) > 0:
		ref_x, ref_y = vertices[-1][0], vertices[-1][1]

		x1, y1 = points.pop(0)
		x2, y2 = points.pop(0)
		x, y = points.pop(0)

		codes += [path.Path.CURVE4] * 3
		vertices.append([x1 + ref_x, y1 + ref_y])
		vertices.append([x2 + ref_x, y2 + ref_y])
		vertices.append([x + ref_x, y + ref_y])

	vertices = numpy.array(vertices)
	vertices[:, 0] = (vertices[:, 0] - xn90) / xscale - 90
	vertices[:, 1] = (vertices[:, 1] - y0) / yscale

	return vertices, codes

def plotRotation(fname, Sx, Sy, Sz):
	"""Plot spin tomography figure"""

	n = 300
	N = 1200.0

	# angles for tomography
	angles = numpy.arange(n + 1) / float(n) * 180 - 90
	angles_radian = angles * 2 * numpy.pi / 360

	# Calculate \Delta^2 \hat{S}_\theta
	ca = numpy.cos(angles_radian)
	sa = numpy.sin(angles_radian)
	d2S = (Sz ** 2).mean() * ca ** 2 + (Sy ** 2).mean() * sa ** 2 - \
		2 * (Sz * Sy).mean() * sa * ca - Sy.mean() ** 2 * sa ** 2 - Sz.mean() ** 2 * ca ** 2 + \
		2 * Sz.mean() * Sy.mean() * sa * ca
	res = d2S / N * 4

	vertices, codes = buildRiedelTomographyPath()
	riedel_path = path.Path(vertices, codes)
	patch = patches.PathPatch(riedel_path, edgecolor='blue', facecolor='none', linestyle='dashed')

	updateParams()
	fig = plt.figure()

	a = 0.19
	b = 0.23
	axes = [a, b, 0.95-a, 0.96-b]
	subplot = fig.add_axes(axes)
	subplot.plot(angles, numpy.log10(res) * 10, 'r')
	subplot.add_patch(patch)
	subplot.set_xlim(xmin=-90, xmax=90)
	subplot.set_ylim(ymin=-13, ymax=20)
	subplot.xaxis.set_ticks((-90, -45, 0, 45, 90))
	subplot.xaxis.set_ticklabels(('-90', '-45', '0', '45', '90'))
	subplot.set_xlabel('Turning angle, $\\theta$ (degrees)')
	subplot.set_ylabel('$\\Delta \\hat{S}_\\theta^2 / (N / 4)$ (dB)')
	fig.savefig(fname)


if __name__ == '__main__':

	updateParams()

	ramsey_visibility_gpe = json.load(open('data/long_time_ramsey/ramsey_gpe_vis.json'))
	ramsey_visibility_qn = json.load(open('data/long_time_ramsey/ramsey_wigner_vis.json'))
	plotXYGraph(
		[ramsey_visibility_gpe, ramsey_visibility_qn],
		['r--', 'b-'],
		'long_ramsey_visibility.eps',
		xmin=0, xmax=5.0, ymin=0, ymax=1.05)

	ramsey_visibility_gpe = json.load(open('data/long_time_rephasing/rephasing_gpe_vis.json'))
	ramsey_visibility_qn = json.load(open('data/long_time_rephasing/rephasing_wigner_vis.json'))
	plotXYGraph(
		[ramsey_visibility_gpe, ramsey_visibility_qn],
		['r--', 'b-'],
		'long_rephasing_visibility.eps',
		xmin=0, xmax=5.0, ymin=0, ymax=1.05)

	ramsey_squeezing_qn_80 = json.load(open('data/squeezing/squeezing_ramsey_80.0.json'))
	ramsey_squeezing_qn_85 = json.load(open('data/squeezing/squeezing_ramsey_85.0.json'))
	ramsey_squeezing_qn_90 = json.load(open('data/squeezing/squeezing_ramsey_90.0.json'))
	ramsey_squeezing_qn_95 = json.load(open('data/squeezing/squeezing_ramsey_95.0.json'))
	plotXYGraph(
		[ramsey_squeezing_qn_80, ramsey_squeezing_qn_85, ramsey_squeezing_qn_90, ramsey_squeezing_qn_95],
		['b-', 'r--', 'g-.', 'k:'],
		'ramsey_squeezing.eps',
		xmin=0, xmax=100.0, ymin=-7.0, ymax=1.0)


	spins = pickle.load(open('data/riedel_comparison/split_potentials_spins_last.pickle'))
	Sx = spins['Sx']
	Sy = -spins['Sy'] # for some reason Y direction in Riedel is swapped (is it the sign of detuning?)
	Sz = spins['Sz']

	plotRotation('riedel_rotation.eps', Sx, Sy, Sz)

	Sx -= Sx.mean()
	Sy -= Sy.mean()
	Sz -= Sz.mean()
#	plot3DView('riedel_cloud_3d.eps', Sx, Sy, Sz)
	plotMainView('riedel_cloud_yz.eps', Sx, Sy, Sz)