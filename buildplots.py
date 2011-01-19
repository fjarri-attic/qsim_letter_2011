import numpy
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def updateParams():
	fig_width_pt = 0.7 * 246.0
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

def plotXYGraph(datasets, linetypes, name):

	fig = plt.figure()

	a = 0.16
	b = 0.23
	axes = [a, b, 0.98-a, 0.97-b]

	subplot = fig.add_axes(axes, xlabel=datasets[0]['xname'], ylabel=datasets[0]['yname'])

	subplot.set_xlim(
		xmin=numpy.array(datasets[0]['xarray']).min(),
		xmax=numpy.array(datasets[0]['xarray']).max())
	subplot.set_ylim(ymin=datasets[0]['ymin'], ymax=datasets[0]['ymax'] + 0.05)

	for dataset, linetype in zip(datasets, linetypes):

		if 'yerrors' not in dataset:
			subplot.plot(
				numpy.array(dataset['xarray']),
				numpy.array(dataset['yarray']),
				linetype, label=dataset['name'])
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

	ramsey_visibility_gpe = json.load(open('data/ramsey_visibility/ramsey_visibility_gpe.json'))
	ramsey_visibility_qn = json.load(open('data/ramsey_visibility/ramsey_visibility_qn.json'))
	ramsey_visibility_exp = json.load(open('data/ramsey_visibility/ramsey_visibility_exp.json'))
	plotXYGraph(
		[ramsey_visibility_gpe, ramsey_visibility_qn, ramsey_visibility_exp],
		['r--', 'b-', 'k.'],
		'figures/ramsey_visibility.eps')

	echo_visibility_gpe = json.load(open('data/echo_visibility/echo_visibility_gpe.json'))
	echo_visibility_qn = json.load(open('data/echo_visibility/echo_visibility_qn.json'))
	echo_visibility_exp = json.load(open('data/echo_visibility/echo_visibility_exp.json'))
	plotXYGraph(
		[echo_visibility_gpe, echo_visibility_qn, echo_visibility_exp],
		['r--', 'b-', 'k.'],
		'figures/echo_visibility.eps')
