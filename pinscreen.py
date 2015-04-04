#!/usr/bin/env python

import argparse
from copy import deepcopy
import os
import sys

import scipy as sp
import matplotlib as mpl
if __name__ == "__main__": mpl.use('Agg')  # we need to do this right away
import numpy as np
from numpy import pi, sin, floor, copysign, sqrt
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.signal import sawtooth


sign = lambda x: copysign(1, x)


class Dot:
    "Simple class to hold an (x,y) pair."
    def __init__(self, xpos, ypos, perim):
        self.xpos = xpos
        self.ypos = ypos
        self.perim = perim

    def __repr__(self):
        return 'Dot(xpos=%f, ypos=%f, perim=%f)' % (self.xpos, self.ypos, self.perim)


class SineFit:
    "Stores parameters for an arbitrary sine function."
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
        "SineFit.eval(self,t): Evaluate the sine function represented by self at a point or list of points t."
        singleton = not getattr(t, '__iter__', False)
        if singleton: t = [t]
        ret = [self.amplitude * sawtooth(2*pi/self.period * ti + self.phase, width=0.5) + self.offset for ti in t]
        if singleton: return ret[0]
        return ret

    def __repr__(self):
        l = []
        l.append('SineFit(amplitude=%f, period=%f, phase=%f, offset=%f, r2=' % (self.amplitude, self.period, self.phase, self.offset))
        if self.r2 is None: l.append('None)')
        else: l.append('%f)' % self.r2)
        return ''.join(l)


def parse_mtrack2(fileobj):
    """frames = parse_mtrack2(fileobj)
    Reads in output from ImageJ plugin MTrack2. Converts it into a list of
    lists of Dot objects:
    frames[0] = [Dot0, Dot1,...].
    Reorganizes the dots so that 0, 1, 2, ... sqrt(n) is the top row from L to
    R, thence onwards."""
    headers = fileobj.readline()[:-1].split('\t')
    n = (len(headers)-1)/3
    if abs(sqrt(n) - int(sqrt(n))) > 0.01:
        raise "Number of dots does not describe a square."
    # x_col[1] is the index in line for X1
    x_col, y_col = [1 + i*3 for i in xrange(n)], [2 + i*3 for i in xrange(n)]
    assignments = None
    fileobj.readline() # discard the line "Tracks 1 to n"
    frames = []
    for line in fileobj.readlines():
        line = line[:-1].split('\t')
        if not assignments:
            # MTrack2 does not guarantee that the dots will be enumerated in any particular order,
            # so we have to figure out which dot in the file is our dot 1. We do this by sorting
            # the dots in the file by both x and y. For an n x n matrix, if a dot has one of the
            # n lowest x values, it must be in the first column; if it's not in the first n but
            # is in the first 2n, it must be in the second column, and so on.
            x, y = ([(i, float(line[col])) for i, col in enumerate(x_col)],
                    [(i, float(line[col])) for i, col in enumerate(y_col)])
            x = sorted(x, cmp=lambda a,b: cmp(a[1], b[1]))
            y = sorted(y, cmp=lambda a,b: cmp(a[1], b[1]))
            xi, yi = [None]*n, [None]*n
            for sort_i, (file_i, value) in enumerate(x): #@UnusedVariable
                xi[file_i] = sort_i
            for sort_i, (file_i, value) in enumerate(y): #@UnusedVariable
                yi[file_i] = sort_i
            assignments = [None] * n
            for i in xrange(n):
                row = int(floor(yi[i] / int(sqrt(n))))
                col = int(floor(xi[i] / int(sqrt(n))))
                assignments[i] = row*int(sqrt(n)) + col
        frame = [None]*n
        for i in xrange(n):
            frame[assignments[i]] = Dot(float(line[x_col[i]]), float(line[y_col[i]]), 1)
        frames.append(frame)
    print [i for i in enumerate(assignments)]
    return frames


def sinefit(frames, dt = 1.0/30.0):
    """fit_parameters = sinefit(frames)
    Takes the output of parse_csv and runs a sine-fitting function against it.
    For frames with n dots, returns an n-element list of tuples of SineFit objects (x,y).
    e.g., fit_parameters[0] = (SineFit(for dot0 in x), SineFit(for dot0 in y))
    """

    # p = [amplitude, period, phase offset, y offset]
    fitfunc = lambda p, x: p[0] * sawtooth(2*pi/p[1]*x + p[2], width=0.5) + p[3]
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    p0 = [1., 1., 0., 0.]
    t = np.arange(len(frames)) * dt
    fit_parameters = []

    for idot in xrange(len(frames[0])):
        print 'Sine fitting: dot %d' % idot
        dx, dy = zip(*[(frame[idot].xpos, frame[idot].ypos) for frame in frames])
        p0[0] = (max(dx)-min(dx))/2.0
        p0[3] = np.mean(dx)
        # FIXME: "success" here is not a valid success measure
        px, success = optimize.leastsq(errfunc, p0, args=(t, dx))
        if not success:
            raise "Problem with optimize for dot %d in x" % idot
        xfit = SineFit(*px)
        xfit.r2 = stats.mstats.pearsonr(dx, xfit.eval(t))[0] ** 2
        p0[0] = (max(dy)-min(dy))/2.0
        p0[3] = np.mean(dy)
        py, success = optimize.leastsq(errfunc, p0, args=(t, dy))
        if not success:
            raise "Problem with optimize for dot %d in y" % idot
        yfit = SineFit(*py)
        yfit.r2 = stats.mstats.pearsonr(dy, yfit.eval(t))[0] ** 2
        fit_parameters.append( (xfit, yfit) )

    return fit_parameters

def process_coordinates(fit_parameters):
    """(center_x, center_y, resting_x, resting_y, extended_x, extended_y) = process_coordinates(fit_parameters)
    finds the resting and extended position for each dot, using the sine fit parameters."""
    # start by finding a coordinate system based on the center of the device.
    # assume the outer dots make a perfect square and (0,0) is upper left.
    X, Y = 0, 1
    center_x = ((fit_parameters[-1][0].offset-fit_parameters[-1][0].amplitude) - (fit_parameters[0][0].offset+fit_parameters[0][0].amplitude))/2+fit_parameters[0][0].offset
    center_y = ((fit_parameters[-1][1].offset-fit_parameters[-1][1].amplitude) - (fit_parameters[0][1].offset+fit_parameters[0][1].amplitude))+fit_parameters[0][1].offset
    #displacement_sign_x = [sign(dot[X].offset-center_x) for dot in fit_parameters] # negative left of center, positive right of center
    #displacement_sign_y = [sign(dot[Y].offset-center_y) for dot in fit_parameters] # negative above center, positive below center

    # resting positions fall when the x coordinate is closest to the center
    # we want: resting posistions to fall when the x coordinate is furthest from center
    #extended_x = [dot[X].offset - displacement_sign_x[di]*dot[X].amplitude for (di, dot) in enumerate(fit_parameters)]
    #extended_y =  [yfit.eval(xfit.period/(2*pi) * ((1-displacement_sign_x[di])*pi/4.0 - xfit.phase)) for (di, (xfit, yfit)) in enumerate(fit_parameters)]

    resting_y = [dot[Y].offset + dot[Y].amplitude for dot in fit_parameters]
    resting_x = [xfit.eval(yfit.period/(2*pi) * (pi/2 - yfit.phase)) for (xfit, yfit) in fit_parameters]

    # extended positions fall when the x coordinate is furthest from the center
    # we want: extended positions to fell when the
    extended_y = [dot[Y].offset - dot[Y].amplitude for dot in fit_parameters]
    extended_x = [xfit.eval(yfit.period/(2*pi) * (3*pi/2 - yfit.phase)) for (xfit, yfit) in fit_parameters]
    # resting_x = [xfit.eval(2*pi*yfit.phase / yfit.period) for (xfit, yfit) in fit_parameters]

    #resting_x = [dot[X].offset + displacement_sign_x[di]*dot[X].amplitude for (di, dot) in enumerate(fit_parameters)]
    # resting_y = [yfit.eval(xfit.period/(2*pi) * ((1+displacement_sign_x[di])*pi/4.0 - xfit.phase)) for (di, (xfit, yfit)) in enumerate(fit_parameters)]
    return (center_x, center_y, resting_x, resting_y, extended_x, extended_y)

def find_center_by_frame(frames):
    return [(np.mean([dot.xpos for dot in frame]), np.mean([dot.ypos for dot in frame])) for frame in frames]

def recenter(frames):
    """frames, [[residuals_x, residuals_y]] = recenter(frames)
    If the center of the device moved while your movie was running, that would
    be bad. But if you were providing a symmetric stretch, we can correct for
    it. This function makes sure that all frames have the same center point.
    It works on the assumption that (mean(x), mean(y)) is always the true
    center of the device. It computes an x offset and y offset value for each
    frame and then adds the same offset to all points within a frame to
    recenter it."""
    frames = deepcopy(frames)
    center_by_frame = find_center_by_frame(frames)
    center_x, center_y = center_by_frame[0]
    centers_x, centers_y = zip(*center_by_frame)
    # show displacement of center from center_x, _y
    centers_x = [frame_pos - center_x for frame_pos in centers_x]
    centers_y = [frame_pos - center_y for frame_pos in centers_y]
    # add dx, dy to each dot in each frame
    for i, frame in enumerate(frames):
        for dot in frame:
            dot.xpos -= centers_x[i]
            dot.ypos -= centers_y[i]
    return frames, [centers_x, centers_y]

def calculate_peak_strain(fit_parameters, resting_x, resting_y, extended_x, extended_y):
    # calculate strain at peak extension
    n_dots = len(fit_parameters)
    strain_x = []
    strain_y = []

    # now, compute the discrete derivative in each axis
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

def write_plots(frames, fit_parameters, jitter, directory, min_strain=-0.1, max_strain=0.3, dt=1/30.0):
    # draw residual plots for each sine fit in x and y
    t = np.arange(len(frames)) * dt
    fit = lambda t, sf: sf.eval(t)

    # TODO show phase for each regression

    # plot the sine fits first
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
        # plt.legend(['in X', 'fit in X', 'in Y', 'fit in Y'])
        axes = plt.gca()
        axes.text(0.95, 0.5,
                  (r'x: $%.2f sin(\frac{2 \pi}{%.2f} t + %.2f) + %.2f$; $R^2=%.4f$' '\n'
                   r'y: $%.2f sin(\frac{2 \pi}{%.2f} t + %.2f) + %.2f$; $R^2=%.4f$'
                   % (fit_x.amplitude, fit_x.period, fit_x.phase, fit_x.offset, fit_x.r2,
                      fit_y.amplitude, fit_y.period, fit_y.phase, fit_y.offset, fit_y.r2)),
                  verticalalignment='center', horizontalalignment='right', transform=axes.transAxes)
        plt.savefig('%s/dot_%04d_fit.png' % (directory, idot))

    # plot the resting and extended coordinates
    (center_x, center_y, resting_x, resting_y, extended_x, extended_y) = process_coordinates(fit_parameters)
    plt.clf()
    plt.axis([min(extended_x+resting_x)-50, max(extended_x+resting_x)+50,
              max(extended_y+resting_y)+50, min(extended_y+resting_y)-50])
    plt.quiver(resting_x, resting_y,
               [ext-rest for (ext, rest) in zip(extended_x, resting_x)],
               [ext-rest for (ext, rest) in zip(extended_y, resting_y)],
               units='xy', angles='xy', scale=1.0)
    plt.savefig('%s/coordinates.png' % directory)

    # plot coordinate system jitter
    plt.clf()
    plt.plot(t, jitter[0], t, jitter[1])
    plt.legend(['x', 'y'])
    plt.savefig('%s/center_displacement.png' % directory)
    plt.clf()
    center_by_frame = find_center_by_frame(frames)
    plt.plot(t, zip(*center_by_frame)[0], t, zip(*center_by_frame)[1])
    plt.savefig('%s/center_position_post.png' % directory)

    n = int(sqrt(len(fit_parameters)))
    peak_strain = calculate_peak_strain(fit_parameters, resting_x, resting_y, extended_x, extended_y)
    min_strain = min_strain or min(peak_strain[0] + peak_strain[1])
    max_strain = max_strain or max(peak_strain[0] + peak_strain[1])
    matrix = lambda axis: np.array(peak_strain[axis]).reshape(n,n)
    for (axis, label) in [(0, 'x'), (1, 'y')]:
        plt.clf()
        plt.pcolor(matrix(axis), edgecolor='k', vmin=min_strain, vmax=max_strain)
        for i in range(n):
            for j in range(n):
                plt.text(i+0.5,j+0.5,"%.4f" % matrix(axis)[j,i],horizontalalignment='center',verticalalignment='center')
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.colorbar(ticks=[-.05,0,.05,.1,.15,.2,.25])
        plt.savefig('%s/peakstrain_%s.png' % (directory, label))

    f = open('%s/index.html' % directory, 'w')
    print >> f, "<!DOCTYPE html>\n<html><head><title>Regression results</title></head><body>"
    print >> f, '<h1>Dot positions</h1><img src="coordinates.png" />'
    print >> f, ('<h1>Center displacement (pre-correction)</h1>'
                 '<img src="center_displacement.png" />')
    print >> f, ('<h1>Center position (post-correction)</h1>'
                 '<img src="center_position_post.png" />')
    print >> f, ('<h1>Peak strain: x</h1>'
                 '<img src="peakstrain_x.png" />'
                 '<p>Mean peak x strain: %f Standard deviation: %f</p>'
                 % (np.mean(peak_strain[0]), np.std(peak_strain[0])))
    print >> f, ('<h1>Peak strain: y</h1>'
                 '<img src="peakstrain_y.png" />'
                 '<p>Mean peak y strain: %f Standard deviation: %f</p>'
                 % (np.mean(peak_strain[1]), np.std(peak_strain[1])))
    for idot in xrange(len(frames[0])):
        print >> f, '<h1>Dot %d</h1><img src="dot_%04d_fit.png" />' % (idot, idot)
    print >> f, '</body></html>'
    f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outpath')
    parser.add_argument('--overwrite', '-O', action='store_true')
    args = parser.parse_args()
    try:
        os.makedirs(args.outpath)  # do this first so we aren't surprised later
    except OSError:
        if not args.overwrite:
            print >> sys.stderr, "Output path exists. Use --overwrite to run anyway."
            sys.exit(1)
    f = open(args.infile, 'rU')
    frames = parse_mtrack2(f)
    f.close()
    centered_frames, jitter = recenter(frames)
    fit_parameters = sinefit(frames)
    write_plots(frames, fit_parameters, jitter, directory=args.outpath, min_strain=-0.05, max_strain=0.25)

if __name__ == '__main__':
    main()
