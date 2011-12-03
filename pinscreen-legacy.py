import csv
from pinscreen import Dot

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


