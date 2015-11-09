import argparse
import os
import sys
import traceback

import cv2
from cv2 import cv
import matplotlib.pyplot as plt
import numpy as np

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
        help'Weight on the smoothness term of the energy function (default 1.0).')

    return validate_args(parser.parse_args())


def main():
    try:
        args = configure_args()
    except IOError as err:
        return str(err)

    # Load the grayscale image.
    print 'Loading the grayscale target image and calculating superpixels.'
    try:
        grayscale = Image(args.grayscale, color=False)
    except BadImageError:
        return 'There was an error reading the grayscale image.'
    
    # Load the context images.
    print 'Loading the color reference images and calculating superpixels.'
    context = []
    for path in args.context:
        try:
            context.append(Image(path, color=True))
        except BadImageError:
            return 'There was an error loading the image at %s' % path

    # Normalize the grayscale distributiongs
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
    histogram = np.zeros((grayscale.shape[0], grayscale.shape[1], args.num_bins), dtype=float)
    for i in grayscale.shape[0]:
        for j in grayscale.shape[1]:
            for ref in context:
                for r_sp in ref.lookup_normalized(grayscale.raw[i,j]):
                    histogram[i,j,color_distribution.lookup(r_sp).index] += 1
            histogram[i,j,:] /= np.sum(histogram[i,j,:])

    # This is where the real magic happens.
    print 'Building and Solving the Markov Random Field'
    field = MarkovGraph(grayscale, color_distribution, histogram, smoothness=args.smoothness)
    labelled = field.solve()

    # Build the color image from the given labelling.
    print 'Generating the Color Image'
    result = np.zeros((grayscale.shape[0], grayscale.shape[1], 3), dtype=np.uint8)
    np.copyto(result[...,0], grayscale.raw)
    for i in range(labelled.shape[0]):
        for j in range(labelled.shape[1]):
            color = np.array(color_distribution.bins[labelled[i,j]].average_color).astype(np.uint8)
            np.copyto(result[i,j,1:], color)

    # Produce the final color RGB image.
    final_image = cv2.cvtColor(result, cv.CV_Lab2RGB)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('\nReceived keyboard interrupt, aborting...')
    except Exception:
        sys.exit(traceback.format_exc())
