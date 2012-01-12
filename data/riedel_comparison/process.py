import math
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

import numpy

xsize = 100.0
ysize = 250.0
zsize = 50.0
zbins = 40


def plotView(Sx, Sy, Sz):
	nullfmt = NullFormatter()
	fig_width = 8

	xbins = int(zbins * xsize / zsize)
	ybins = int(zbins * ysize / zsize)

	x_d = 0.12
	x_dd = 0.03
	x_ly = (1.0 - x_d - x_dd * 2) / (1.0 + zsize / xsize)
	x_lx = (1.0 - x_d - x_dd * 2) / (1.0 + xsize / zsize)

	y_d = x_d
	y_dd = x_dd
	y_lx = x_lx
	y_lz = y_lx / xsize * zsize

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
	plt.figure(1, figsize=(fig_width, fig_width / aspect))

	axYZ = plt.axes(rectYZ)
	axXY = plt.axes(rectXY)
	axXZ = plt.axes(rectXZ)

	# no labels
	axXY.xaxis.set_major_formatter(nullfmt)
	axXZ.yaxis.set_major_formatter(nullfmt)

	axYZ.hexbin(Sy, Sz, cmap=cm.jet, gridsize=(ybins, zbins))
	axYZ.set_xlabel('$S_y$')
	axYZ.set_ylabel('$S_z$')
	axYZ.set_xlim(xmin=-ysize, xmax=ysize)
	axYZ.set_ylim(ymin=-zsize, ymax=zsize)

	axXY.hexbin(Sy, Sx, cmap=cm.jet, gridsize=(ybins, xbins))
	axXY.set_ylabel('$S_x$')
	axXY.set_xlim(xmin=-ysize, xmax=ysize)
	axXY.set_ylim(ymin=-xsize, ymax=xsize)

	axXZ.hexbin(Sx, Sz, cmap=cm.jet, gridsize=(xbins, zbins))
	axXZ.set_xlabel('$S_x$')
	axXZ.set_xlim(xmin=-xsize, xmax=xsize)
	axXZ.set_ylim(ymin=-zsize, ymax=zsize)

	plt.savefig('cloud_3dview.pdf')

def plotMainView(Sx, Sy, Sz):

	ybins = int(float(zbins) * ysize / zsize + 0.5)

	updateParams(aspect=0.3)
	fig = plt.figure()

	#	subplot = fig.add_subplot(111, aspect=1)
	a = 0.19
	b = 0.23
	axes = [a, b, 0.95-a, 0.96-b]
	subplot = fig.add_axes(axes, aspect=1)

	min_angle = 9.6 / 180 * numpy.pi
	min_var = numpy.sqrt(25.349280804)

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

	r = numpy.array([0, 1.0])
	l1_x = (r * 2 - 1) * min_var * numpy.sin(min_angle)
	l1_y = (-r * 2 + 1) * min_var * numpy.cos(min_angle)

	l2_x = r * 150
	l2_y = r * 0

	l3_x = r * 150
	l3_y = r * 150 * numpy.sin(min_angle)

#	subplot.hexbin(Sy, Sz, cmap=cm.PuRd, gridsize=(ybins, zbins))
	hist,xedges,yedges = numpy.histogram2d(Sy,Sz,bins=(ybins, zbins),range=[[-250,250],[-50,50]])
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
	subplot.imshow(hist.T, extent=extent, interpolation='nearest', origin='lower', cmap=cm.PuRd)

	subplot.arrow(arrow1_x, arrow1_y, arrow1_dx, arrow1_dy, **arrow_kwds)
	subplot.arrow(-arrow1_x, -arrow1_y, -arrow1_dx, -arrow1_dy, **arrow_kwds)

	subplot.plot(l1_x, l1_y, color='blue', linewidth=0.3)
	subplot.plot(l2_x, l2_y, linestyle='--', color='black', linewidth=0.3, dashes=(6, 2))
	subplot.plot(l3_x, l3_y, linestyle='--', color='black', linewidth=0.3, dashes=(6, 2))

	arc = matplotlib.patches.Arc((0.0, 0.0), 100, 100,
		theta1=0, theta2=min_angle / numpy.pi * 180, linewidth=0.3, fill=False)
	subplot.add_patch(arc)

	subplot.text(-30, 20, "$d_\\theta$", fontsize=7)
	subplot.text(40, 10, "$\\theta$", fontsize=7)

	subplot.set_xlabel('$S_y$')
	subplot.set_ylabel('$S_z$')
	subplot.set_xlim(xmin=-ysize, xmax=ysize)
	subplot.set_ylim(ymin=-zsize, ymax=zsize)
	plt.savefig('riedel_cloud_yz.eps')

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

def plotRotation(Sx, Sy, Sz, N):
	n = 300

	angles = numpy.arange(n + 1) / float(n) * 180 - 90
	angles_radian = angles * 2 * numpy.pi / 360

	min_angle = 9.6 / 180 * numpy.pi
	nSz = -numpy.sin(min_angle) * Sy + numpy.cos(min_angle) * Sz
	print nSz.var()

	ca = numpy.cos(angles_radian)
	sa = numpy.sin(angles_radian)
	d2S = (Sz ** 2).mean() * ca ** 2 + (Sy ** 2).mean() * sa ** 2 - \
		2 * (Sz * Sy).mean() * sa * ca - Sy.mean() ** 2 * sa ** 2 - Sz.mean() ** 2 * ca ** 2 + \
		2 * Sz.mean() * Sy.mean() * sa * ca
	res = d2S / N * 4
	print res.min() * N / 4

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

	riedel_path = path.Path(vertices, codes)
	patch = patches.PathPatch(riedel_path, edgecolor='blue', facecolor='none', linestyle='dashed')

	updateParams()
	fig = plt.figure()

	a = 0.19
	b = 0.23
	axes = [a, b, 0.95-a, 0.96-b]
	subplot = fig.add_axes(axes)

#	subplot = fig.add_subplot(111)
	subplot.plot(angles, numpy.log10(res) * 10, 'r')
	subplot.add_patch(patch)
	subplot.set_xlim(xmin=-90, xmax=90)
	subplot.set_ylim(ymin=-13, ymax=20)
	subplot.xaxis.set_ticks((-90, -45, 0, 45, 90))
	subplot.xaxis.set_ticklabels(('-90', '-45', '0', '45', '90'))
	subplot.set_xlabel('Turning angle, $\\theta$ (degrees)')
	subplot.set_ylabel('$\\Delta \\hat{S}_\\theta^2 / (N / 4)$ (dB)')
	plt.savefig('riedel_rotation.eps')


if __name__ == '__main__':

	d = pickle.load(open('split_potentials_spins_last.pickle'))

	Sx = d['Sx']
	Sy = -d['Sy'] # for some reason Y direction in Riedel is swapped (is it the sign of detuning?)
	Sz = d['Sz']

	S = [Sx.mean(), Sy.mean(), Sz.mean()]
	N = numpy.sqrt(S[0] ** 2 + S[1] ** 2 + S[2] ** 2)

	plotRotation(Sx, Sy, Sz, N)

	Sx -= S[0]
	Sy -= S[1]
	Sz -= S[2]

	#	plotView(Sx, Sy, Sz)
	plotMainView(Sx, Sy, Sz)
