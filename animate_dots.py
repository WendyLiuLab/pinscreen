#!/usr/bin/env python

import sys
import md5
import scipy as sp
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from pinscreen import parse_mtrack2, sinefit, process_coordinates, recenter
from math import pi

def colormap(n):
    return ['#'+md5.md5(str(i)).hexdigest()[:6] for i in range(n)]

def main(dotfile):
    frames = parse_mtrack2(open(dotfile, 'rU'))
    colors = colormap(len(frames[0]))
    frames, jitter = recenter(frames)
    fit_parameters = sinefit(frames)
    (center_x, center_y, resting_x, resting_y, extended_x, extended_y) = process_coordinates(fit_parameters)

    plt.ion()
    #plt.figure()
    # axes = plt.axes()
    # plt.axis([0,100,100,0])
    #print "Press Enter when you're ready to go."
    #raw_input()
    while 1:
        for i, frame in enumerate(frames):
            plt.cla()
            plt.axis([0,100,100,0])
            x, y = zip(*[(dot.xpos, dot.ypos) for dot in frame])
            area = [(dot.perim/(2*pi))**2*pi*5 for dot in frame]
            plt.scatter(resting_x, resting_y, c='b')
            plt.scatter(extended_x, extended_y, c='r')
            plt.scatter(x, y, area, c='g') #colors)
            plt.plot()
            plt.draw()
            print i

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'usage: animate_dots.py dots.csv'
        sys.exit(1)
    main(sys.argv[1])

