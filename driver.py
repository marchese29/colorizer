import argparse
import os
import sys
import traceback

import cv2
from cv2 import cv
import numpy as np

from cvutil.images import Image, BadImageError


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

    return validate_args(parser.parse_args())


def main():
    try:
        args = configure_args()
    except IOError as err:
        return str(err)

    # Load the grayscale image.
    try:
        grayscale = Image(args.grayscale, color=False)
    except BadImageError:
        return 'There was an error reading the grayscale image.'
    
    # Load the context images.
    context = []
    for path in args.context:
        try:
            context.append(Image(path, color=True))
        except BadImageError:
            return 'There was an error loading the image at %s' % path

    # Normalize the grayscale histograms of the context images to the target image's histogram.
    pass

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('\nReceived keyboard interrupt, aborting...')
    except Exception as e:
        sys.exit(traceback.format_exc())
