import argparse
import os
import sys
import traceback

import cv2
from cv2 import cv
import matplotlib.pyplot as plt
import numpy as np
from progressbar import Bar, ETA, Percentage, ProgressBar

from cvutil.distributions import LabColorDistribution
from cvutil.images import Image, BadImageError
from cvutil.markov import MarkovGraph


def validate_args(args):
    # Validate that the grayscale image is a valid file.
    args.grayscale = os.path.abspath(os.path.expanduser(args.grayscale))
    if not os.path.isfile(args.grayscale):
        raise IOError('File not found: %s' % args.grayscale)

    # Validate that the color images are valid files.
    context_images = []
    for location in args.context:
        path = os.path.abspath(os.path.expanduser(location))
        if not os.path.isfile(path):
            raise IOError('File not found: %s' % path)
        context_images.append(path)
    args.context = context_images

    return args


def configure_args():
    parser = argparse.ArgumentParser(
        description='Image colorization using the context composition algorithm.')

    parser.add_argument('grayscale', help='File location of the grayscale image to be colorized.')
    parser.add_argument('context', nargs='+', help='File locations of the color context images.')

    parser.add_argument('--num-bins', type=int, default=75,
        help='Number of color bins to use (default 75).')
    parser.add_argument('--smoothness', type=float, default=1.0,
        help='Weight on the smoothness term of the energy function (default 1.0).')
    parser.add_argument('--feature', choices={'luminance', 'variance-freq', 'variance-width'},
        default='luminance', help='Which feature of the image to use in colorization.')
    parser.add_argument('--save-image', action='store_true',
        help='Saves the resulting image as result.jpg in the current directory.')

    return validate_args(parser.parse_args())


def main():
    try:
        args = configure_args()
    except IOError as err:
        return str(err)

    # Load the grayscale image.
    print 'Loading the grayscale target image and calculating neighborhoods.'
    try:
        grayscale = Image(args.grayscale, color=False, stat=args.feature)
    except BadImageError:
        return 'There was an error reading the grayscale image.'
    
    # Load the context images.
    print 'Loading the color reference images and calculating neighborhoods.'
    context = []
    for path in args.context:
        try:
            context.append(Image(path, color=True, stat=args.feature))
        except BadImageError:
            return 'There was an error loading the image at %s' % path

    # Normalize the grayscale distributions if we need to.
    if args.feature == 'luminance':
        print 'Normalizing the reference image grayscale distributions'
        for ref in context:
            ref.configure_normalized_distribution(grayscale)

    # Generate the Lab Color Distribution
    print 'Generating the color distribution.'
    color_distribution = LabColorDistribution(context, num_bins=args.num_bins)
    for idx, b in enumerate(color_distribution):
        b.set_index(idx)

    # Generate the color probability histogram for each superpixel in the target.
    print 'Generating color probability distributions.'
    if args.feature == 'luminance':
        # Our probability distribution is defined by the number of superpixels from the
        # corresponding grayscale bin in each reference image.
        widgets = ['Generating: ', Percentage(), Bar(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=np.size(grayscale.raw)).start()
        histogram = np.zeros((grayscale.shape[0], grayscale.shape[1], args.num_bins), dtype=float)
        pidx = 0
        for i in xrange(grayscale.shape[0]):
            for j in xrange(grayscale.shape[1]):
                for ref in context:
                    for r_sp in ref.lookup_normalized(grayscale.raw[i,j]):
                        histogram[i,j,color_distribution.lookup(r_sp).index] += 1
                s = histogram[i,j,:].sum()
                if s > 1.0:
                    histogram[i,j,:] /= s
                pidx += 1
                pbar.update(pidx)
        pbar.finish()
    elif args.feature == 'variance-freq' or args.feature == 'variance-width':
        # Our probability distribution is defined by the number of windows from the corresponding
        # variance bin in each reference image.
        widgets = ['Generating: ', Percentage(), Bar(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=len(grayscale)).start()
        histogram = np.zeros((grayscale.shape[0]-4, grayscale.shape[1]-4, args.num_bins), dtype=float)
        pidx = 0
        for window in grayscale:
            (i, j) = window.index
            for ref in context:
                winds = ref.lookup_variance(window.variance)
                for r_wind in winds:
                    histogram[i,j,color_distribution.lookup(r_wind).index] += 1
            s = histogram[i,j,:].sum()
            if s > 1.0:
                histogram[i,j,:] /= s
            pbar.update(pidx)
            pidx += 1
        pbar.finish()

    # This is where the real magic happens.
    print 'Building and Solving the Markov Random Field'
    field = MarkovGraph(grayscale.raw[2:-2,2:-2], color_distribution, histogram,
        smoothness=args.smoothness)
    labelled = field.solve()

    # Build the color image from the given labelling.
    print 'Generating the Color Image'
    # We need to clip the image when using certain features.
    if args.feature == 'luminance':
        result = np.zeros((grayscale.shape[0], grayscale.shape[1], 3), dtype=np.uint8)
        np.copyto(result[...,0], grayscale.raw)
    elif args.feature == 'variance-freq' or args.feature == 'variance-width':
        result = np.zeros((grayscale.shape[0]-4, grayscale.shape[1]-4, 3), dtype=np.uint8)
        np.copyto(result[...,0], grayscale.raw[2:-2, 2:-2])

    # Recover the color from each label
    for i in range(labelled.shape[0]):
        for j in range(labelled.shape[1]):
            color = np.array(color_distribution.bins[labelled[i,j]].average_color).astype(np.uint8)
            result[i,j,1:] = color

    # Produce the final color RGB image.
    final_image = cv2.cvtColor(result, cv.CV_Lab2RGB)

    # Display the resulting image.
    fig = plt.figure('Colorized Image')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(final_image)
    plt.axis('off')
    plt.show()

    if args.save_image:
        cv2.imwrite('result.jpg', final_image)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('\nReceived keyboard interrupt, aborting...')
    except Exception:
        sys.exit(traceback.format_exc())
