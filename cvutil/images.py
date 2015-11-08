'''
Provides a utility for loading images with several important up-front data extractions.
Images in this library can be iterated over:
    for superpixel in image:
        <Use the Superpixel>

Each superpixel has a median_color property (for color images), and a median_intensity property.

To retrieve the raw data extracted using OpenCV (in RGB form for color images), use the image's
'raw' property.  Note that the use of the raw underlying image is heavily discouraged.

Additionally, one can display the image using its 'display' function.  To show the superpixels on
top of the image, simply pass the keyword argument: superpixels=True.
'''
import cv2
from cv2 import cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic

from util import round5_down

class BadImageError(Exception):
    '''Occurs when an image is not loaded correctly by OpenCV.'''
    pass

class NotColorError(AttributeError):
    '''Occurs when trying to perform color operations on a grayscale image.'''
    pass

class Image(object):
    '''Represents a single image.'''

    class Superpixel(object):
        '''Represents a single superpixel.'''
        def __init__(self, image, indices, id):
            '''Stores a reference to the containing image, as well as calculates the median
            properties.
            '''
            self._id = id
            self._image = image
            self._indices = indices
            self._coordinates = np.nonzero(indices)
            self._size = np.count_nonzero(indices)

            # Calculate the average color for this pixel
            if image._color:
                lumin = np.median(self._image._labimage[:,:,0][indices].astype(np.int64))
                alpha = np.median(self._image._labimage[:,:,1][indices].astype(np.int64))
                beta  = np.median(self._image._labimage[:,:,2][indices].astype(np.int64))
                self._median_color = (alpha, beta)
            else:
                lumin = np.median(self._image._raw[indices].astype(np.int64))

            self._median_intensity = lumin

        def __eq__(self, other):
            return self._id == other._id and self._image is other._image

        @property
        def median_color(self):
            '''The median color for this superpixel represented as a tuple of floats: (alpha, beta).
            '''
            if self._image._color:
                return self._median_color
            else:
                raise NotColorError('Attempted color access of grayscale superpixel.')

        @property
        def median_intensity(self):
            '''The median intensity for this superpixel.'''
            return self._median_intensity

    def _configure_superpixels(self):
        '''Configures the superpixels for this image.'''
        num_pixels = self._raw.shape[0] * self._raw.shape[1]
        if self._color:
            self._segments = slic(self._raw, convert2lab=True, n_segments=num_pixels / 50,
                enforce_connectivity=True)
        else:
            self._segments = slic(self._raw, n_segments=num_pixels / 50, compactness=0.1,
                enforce_connectivity=True)

        # Create the superpixel objects.
        self._superpixels = []
        for i in xrange(np.max(self._segments)):
            indices = (self._segments == i)
            self._superpixels.append(Image.Superpixel(self, indices, i))

    def _generate_grayscale_histogram(self):
        '''Configures the histogram for this image.'''
        self._histogram, _ = np.histogram([sp.median_intensity for sp in self], range=(0, 255),
                                          bins=51, density=True)
        self._distribution = { \
            i: [p for p in self if round5_down(p.median_intensity) == i] for i in range(0, 255, 5) \
        }

    def __init__(self, path, color=False):
        '''Loads the image at the given path.'''
        self._path = path
        self._color = color
        if self._color:
            self._raw = cv2.cvtColor(cv2.imread(path, cv.CV_LOAD_IMAGE_COLOR), cv.CV_BGR2RGB)
        else:
            self._raw = cv2.imread(path, cv.CV_LOAD_IMAGE_GRAYSCALE)

        if self._raw is None:
            raise BadImageError()

        if color:
            self._labimage = cv2.cvtColor(self._raw, cv.CV_RGB2Lab)

        # Calculate superpixels
        self._configure_superpixels()

        # Generate the grayscale histogram
        self._generate_grayscale_histogram()

    def __iter__(self):
        '''You can iterate over the superpixels if you would like to.'''
        return iter(self._superpixels)

    def __len__(self):
        '''The "length" of the image is represented by the number of superpixels in it.'''
        return len(self._superpixels)

    @property
    def raw(self):
        '''The raw RGB/L matrix returned by OpenCV (DO NOT MODIFY, UNDEFINED BEHAVIOR).'''
        return self._raw

    @property
    def color(self):
        '''Whether or not this is a color image.'''
        return self._color

    @property
    def grayscale_distribution(self):
        '''The superpixels as they are in their bins.'''
        return self._distribution

    @property
    def histogram(self):
        '''The grayscale distribution of the superpixel median intensities.'''
        return self._histogram * (1.0 / np.sum(self._histogram))

    def configure_normalized_distribution(self, image):
        '''Normalizes this images grayscale histogram.'''
        # Sort out the superpixels in descending order of intensity
        sorted_pixels = sorted(self, key=lambda x: x.median_intensity, reverse=True)
        self._normalized = { i: [] for i in range(0, 254, 5) }

        for idx in range(0, 255, 5):
            density = image.histogram[idx/5]
            for j in range(int(density * len(self))):
                self._normalized[idx].append(sorted_pixels.pop())

        # Stick any leftovers in the last bin.
        while len(sorted_pixels) > 0:
            self._normalized[250].append(sorted_pixels.pop())

    def lookup(self, intensity):
        '''Performs a lookup of the given intensity value on this image's non-normalized grayscale
        distribution and returns all of the superpixels in the corresponding bin.
        '''
        return self._distribution[round5_down(intensity)]

    def lookup_normalized(self, intensity):
        '''Performs a lookup of the given intensity value on this image's normalized grayscale
        distribution and returns all of the superpixels in the corresponding bin.
        '''
        if not hasattr(self, '_normalized'):
            raise AttributeError('No normalized histogram has been produced for this image yet.')
        return self._normalized[round5_down(intensity)]

    def graph_distribution(self, normalized=False):
        '''Plots the grayscale distribution for this image.  If normalized is true, the most recent
        normalization of the histogram is displayed.
        '''
        if normalized:
            assert hasattr(self, '_normalized')
            fig = plt.figure('Normalized histogram')
            ax = fig.add_subplot(1, 1, 1)
            dist = []
            for key, value in self._normalized.iteritems():
                dist += ([key] * len(value))
            ax.hist(dist, range=(0, 255), bins=51)
        else:
            fig = plt.figure('Non-normalized histogram')
            ax = fig.add_subplot(1, 1, 1)
            ax.hist([sp.median_intensity for sp in self], range=(0, 255), bins=51)
        plt.axis('on')
        plt.show()

    def display(self, superpixels=False):
        '''Display this image with or without superpixels in matplotlib.'''
        fig = plt.figure('Image: (%d, %d)' % (self._raw.shape[0], self._raw.shape[1]))
        ax = fig.add_subplot(1, 1, 1)
        if superpixels:
            ax.imshow(mark_boundaries(self._raw, self._segments))
        else:
            ax.imshow(self._raw, cmap=plt.cm.gray, vmin=0, vmax=255)
        plt.axis('off')
        plt.show()
