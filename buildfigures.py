import numpy
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import copy


def updateParams():
	fig_width_pt = 0.9 * 246.0
	inches_per_pt = 1.0 / 72.27               # Convert pt to inches
	golden_mean = (math.sqrt(5) - 1.0) / 2.4         # Aesthetic ratio
	fig_width = fig_width_pt * inches_per_pt  # width in inches
	fig_height = fig_width * golden_mean       # height in inches
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
