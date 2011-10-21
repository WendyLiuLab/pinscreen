#!/usr/bin/env python

import sys, csv, os
import scipy as sp
import numpy as np
import matplotlib as mpl
if __name__ == '__main__': mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize, stats
from math import pi, sin, floor, copysign, sqrt

sign = lambda x: copysign(1, x)

class Dot:
    def __init__(self, xpos, ypos, perim):
        self.xpos = xpos
        self.ypos = ypos
        self.perim = perim

    def __repr__(self):
        return 'Dot(xpos=%f, ypos=%f, perim=%f)' % (self.xpos, self.ypos, self.perim)

class SineFit:
    def __init__(self, amplitude, period, phase, offset, r2 = None):
        self.amplitude = amplitude
        self.period = period
        self.phase = phase
        self.offset = offset
        self.r2 = r2
        self.normalize()

    def normalize(self):
        # negative amplitude = pi phase change
        if self.amplitude < 0:
            self.amplitude *= -1
            self.phase += pi
        # restrict phase to -pi ... pi
        if not (-pi < self.phase and self.phase <= pi):
            self.phase -= floor((self.phase+pi)/(2*pi))*(2*pi)

    def eval(self, t):
        singleton = not getattr(t, '__iter__', False)
        if singleton: t = [t]
        ret = [self.amplitude * sin(2*pi/self.period * ti + self.phase) + self.offset for ti in t]
        if singleton: return ret[0]
        return ret

    def __repr__(self):
        l = []
        l.append('SineFit(amplitude=%f, period=%f, phase=%f, offset=%f, r2=' % (self.amplitude, self.period, self.phase, self.offset))
        if self.r2 is None: l.append('None)')
        else: l.append('%f)' % self.r2)
        return ''.join(l)

def usage():
    print 'pinscreen.py dots.csv plot_output_dir'
    print 'Creates plots as plot_output_dir/dot_1_fit.png...'
    print 'plot_output_dir should not exist before running pinscreen.py.'

def parse_csv(fileobj):
    """frames = parse_csv(filename)

    Reads in ImageJ dot-tracker output of the form:
    stack.tif:<frame>,<xmedian>,<ymedian>,<perimeter>

    Converts it into a list of lists of Dot objects:
    frames[0] = [Dot0, Dot1...]
    """
    last_frame = None
    frames = []
    csv_reader = csv.reader(fileobj)
    csv_reader.next() # discard the first row
    for row in csv_reader:
        if row[0] != last_frame:
            last_frame = row[0]
            frames.append([])
        frames[-1].append( Dot(*[float(i) for i in row[1:]]) )
    # do some mild consistency checking: should have the same number of dots in all frames
    if not all([len(frame) == len(frames[0]) for frame in frames]): raise "Consistency error in input."
    return frames

def sinefit(frames, dt = 1.0/30.0):
    """fit_parameters = sinefit(frames)
    Takes the output of parse_csv and runs a sine-fitting function against it.
    For frames with n dots, returns an n-element list of tuples of SineFit objects (x,y).
    e.g., fit_parameters[0] = (SineFit(for dot0 in x), SineFit(for dot0 in y))
    """

    # p = [amplitude, period, phase offset, y offset]
    fitfunc = lambda p, x: np.array([p[0]*sin(2*pi/p[1]*xi + p[2]) + p[3] for xi in x])
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    p0 = [1., 1., 0., 0.]
    t = np.arange(len(frames)) * dt
    fit_parameters = []

    for idot in xrange(len(frames[0])):
        print 'Sine fitting: dot %d' % idot
        dx, dy = zip(*[(frame[idot].xpos, frame[idot].ypos) for frame in frames])
        # FIXME: "success" here is not a valid success measure
        px, success = optimize.leastsq(errfunc, p0, args=(t, dx))
        if not success:
           raise "Problem with optimize for dot %d in x" % idot
        xfit = SineFit(*px)
        xfit.r2 = stats.mstats.pearsonr(dx, xfit.eval(t))[0] ** 2
        py, success = optimize.leastsq(errfunc, p0, args=(t, dy))
        if not success:
           raise "Problem with optimize for dot %d in y" % idot
        yfit = SineFit(*py)
        yfit.r2 = stats.mstats.pearsonr(dy, yfit.eval(t))[0] ** 2
        fit_parameters.append( (xfit, yfit) )

    return fit_parameters

def censor_outliers(frames, fit_parameters, k = 3, noise=0.1):
    """frames = censor_outliers(frames, fit_parameters, k=3)
    Sometimes we get points way out of band. Remove them with extreme prejudice.
    Deletes points more than k*amplitude away from the DC offset by replacing them with their left neighbor."""
    for frame_idx, frame in enumerate(frames):
        for di, dot in enumerate(frame):
            fit_x, fit_y = fit_parameters[di]
            last_x, last_y = None, None
            if frame_idx == 0:
                last_x, last_y = fit_x.eval(0), fit_y.eval(0)
            else:
                last_dot = frames[frame_idx-1][di]
                last_x, last_y = last_dot.xpos, last_dot.ypos
            if fit_x.amplitude > noise and (dot.xpos < (fit_x.offset - k*fit_x.amplitude) or dot.xpos > (fit_x.offset + k*fit_x.amplitude)):
                print 'identified outlier: frame %d dot %d x' % (frame_idx, di)
                print '  outlier position: %f new position: %f' % (dot.xpos, last_x)
                frames[frame_idx][di].xpos = last_x
            if fit_y.amplitude > noise and (dot.ypos < (fit_y.offset - k*fit_y.amplitude) or dot.ypos > (fit_y.offset + k*fit_y.amplitude)):
                print 'identified outlier: frame %d dot %d y' % (frame_idx, di)
                print '  outlier position: %f new position: %f' % (dot.ypos, last_y)
                frames[frame_idx][di].ypos = last_y
    return frames

def process_coordinates(fit_parameters):
    # (center_x, center_y, resting_x, resting_y, extended_x, extended_y) = process_coordinates(fit_parameters)
    # start, for shits and giggles, by finding a coordinate system based on the center of the device.
    # assume the outer dots make a perfect square and (0,0) is upper left.
    X, Y = 0, 1
    center_x = (fit_parameters[-1][0].offset-fit_parameters[-1][0].amplitude) - (fit_parameters[0][0].offset+fit_parameters[0][0].amplitude)
    center_y = (fit_parameters[-1][1].offset-fit_parameters[-1][1].amplitude) - (fit_parameters[0][1].offset+fit_parameters[0][1].amplitude)
    displacement_sign_x = [sign(dot[X].offset-center_x) for dot in fit_parameters] # negative left of center, positive right of center
    displacement_sign_y = [sign(dot[Y].offset-center_y) for dot in fit_parameters] # negative above center, positive below center

    # resting positions fall when the x coordinate is closest to the center
    resting_x = [dot[X].offset - displacement_sign_x[di]*dot[X].amplitude for (di, dot) in enumerate(fit_parameters)]
    resting_y =  [yfit.eval(xfit.period/(2*pi) * ((1-displacement_sign_x[di])*pi/4.0 - xfit.phase)) for (di, (xfit, yfit)) in enumerate(fit_parameters)]

    # extended positions fall when the x coordinate is furthest from the center
    extended_x = [dot[X].offset + displacement_sign_x[di]*dot[X].amplitude for (di, dot) in enumerate(fit_parameters)]
    extended_y = [yfit.eval(xfit.period/(2*pi) * ((1+displacement_sign_x[di])*pi/4.0 - xfit.phase)) for (di, (xfit, yfit)) in enumerate(fit_parameters)]
    return (center_x, center_y, resting_x, resting_y, extended_x, extended_y)

def find_center_by_frame(frames):
    return [(np.mean([dot.xpos for dot in frame]), np.mean([dot.ypos for dot in frame])) for frame in frames]

def calculate_peak_strain(fit_parameters, resting_x, resting_y, extended_x, extended_y):
    # calculate strain at peak extension
    n_dots = len(fit_parameters)
    strain_x = []
    strain_y = []

    # now, compute the discrete 
    n = int(sqrt(n_dots))
    for di in xrange(n_dots):
        # in x
        if di % n == 0: # left edge
            strain_x.append( (extended_x[di+1] - extended_x[di] - (resting_x[di+1] - resting_x[di]) ) / (resting_x[di+1] - resting_x[di]) )
        elif di % n == (n-1): # right edge
            strain_x.append( (extended_x[di] - extended_x[di-1] - (resting_x[di] - resting_x[di-1]) ) / (resting_x[di] - resting_x[di-1]) )
        else: # in the center
            strain_x.append( (extended_x[di+1] - extended_x[di-1] - (resting_x[di+1] - resting_x[di-1]) ) / (resting_x[di+1] - resting_x[di-1]) )
        # in y
        if di < n: # top row
            strain_y.append( (extended_y[di+n] - extended_y[di] - (resting_y[di+n] - resting_y[di]) ) / (resting_y[di+n] - resting_y[di]) )
        elif di >= n_dots-n: # bottom row
            strain_y.append( (extended_y[di] - extended_y[di-n] - (resting_y[di] - resting_y[di-n]) )  / (resting_y[di] - resting_y[di-n]) )
        else: # in the center
            strain_y.append( (extended_y[di+n] - extended_y[di-n] - (resting_y[di+n] - resting_y[di-n]) ) / (resting_y[di+n] - resting_y[di-n]) )
    return (strain_x, strain_y)

def write_plots(frames, fit_parameters, directory, dt=1/30.0):
    # draw residual plots for each sine fit in x and y
    t = np.arange(len(frames)) * dt
    fit = lambda t, sf: sf.eval(t)
    
    for idot in xrange(len(frames[0])):
        actual_x = [frame[idot].xpos for frame in frames]
        actual_y = [frame[idot].ypos for frame in frames]
        fit_x, fit_y = fit_parameters[idot]
        plt.clf()
        plt.plot(t, actual_x, 'b.', t, fit(t, fit_x), 'b-')
        plt.plot(t, actual_y, 'r.', t, fit(t, fit_y), 'r-')
        plt.title('Dot %d' % idot)
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (px)')
        plt.legend(['in X', 'fit in X', 'in Y', 'fit in Y'])
        axes = plt.gca()
        axes.text(0.95, 0.5, r"""x: $%.2f sin(\frac{2 \pi}{%.2f} t + %.2f) + %.2f$; $R^2=%.4f$
y: $%.2f sin(\frac{2 \pi}{%.2f} t + %.2f) + %.2f$; $R^2=%.4f$""" % (fit_x.amplitude, fit_x.period, fit_x.phase, fit_x.offset, fit_x.r2, fit_y.amplitude, fit_y.period, fit_y.phase, fit_y.offset, fit_y.r2), verticalalignment='center', horizontalalignment='right', transform=axes.transAxes)
        plt.savefig('%s/dot_%04d_fit.png' % (directory, idot))

    # plot the resting and extended coordinates
    (center_x, center_y, resting_x, resting_y, extended_x, extended_y) = process_coordinates(fit_parameters)
    plt.clf()
    plt.axis([0,50,50,0])
    plt.scatter(resting_x, resting_y, c='b')
    plt.scatter(extended_x, extended_y, c='r')
    plt.savefig('%s/coordinates.png' % directory)

    n = int(sqrt(len(fit_parameters)))
    peak_strain = calculate_peak_strain(fit_parameters, resting_x, resting_y, extended_x, extended_y)
    matrix = lambda axis: np.array(peak_strain[axis]).reshape(n,n)
    for (axis, label) in [(0, 'x'), (1, 'y')]:
        plt.clf()
        plt.pcolor(matrix(axis), edgecolor='k')
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])        
        plt.colorbar()
        plt.savefig('%s/peakstrain_%s.png' % (directory, label))

    center_by_frame = find_center_by_frame(frames)
    centers_x, centers_y = zip(*center_by_frame)
    # normalize this so we expect the center to be 0,0
    centers_x = [frame_pos - center_x for frame_pos in centers_x]
    centers_y = [frame_pos - center_y for frame_pos in centers_y]
    plt.clf()
    plt.plot(t, centers_x, t, centers_y)
    plt.legend(['X', 'Y'])
    plt.xlabel('Time (s)')
    plt.ylabel('Mean point displacement from center (px)')
    plt.savefig('%s/center_displacement.png' % directory)

    f = open('%s/index.html' % directory, 'w')
    print >> f, "<!DOCTYPE html>\n<html><head><title>Regression results</title></head><body>"
    print >> f, '<h1>Dot positions</h1><img src="coordinates.png" />'
    print >> f, '<h1>Center displacement</h1><img src="center_displacement.png" />'
    print >> f, '<h1>Peak strain: x</h1><img src="peakstrain_x.png" /><h1>Peak strain: y</h1><img src="peakstrain_y.png" />'
    for idot in xrange(len(frames[0])):
        print >> f, '<h1>Dot %d</h1><img src="dot_%04d_fit.png" />' % (idot, idot)
    print >> f, '</body></html>'
    f.close()

def main(argv):
    # sys.argv = [-, input filename, output directory]
    # output directory should not exist:
    if len(argv) != 3 or os.path.exists(argv[2]):
        usage()
        sys.exit(1)
    os.mkdir(argv[2]) # do this first so we aren't surprised later
    f = open(sys.argv[1], 'rU')
    frames = parse_csv(f)
    f.close()
    fit_parameters = sinefit(frames)
    frames = censor_outliers(frames, fit_parameters)
    fit_parameters = sinefit(frames)
    write_plots(frames, fit_parameters, sys.argv[2])

if __name__ == '__main__':
    main(sys.argv)
