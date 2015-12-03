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
                lumin = np.median(self._image._labimage[...,0][indices].astype(float))
                alpha = np.median(self._image._labimage[...,1][indices].astype(float))
                beta  = np.median(self._image._labimage[...,2][indices].astype(float))
                self._median_color = (alpha, beta)
            else:
                lumin = np.median(self._image._raw[indices].astype(float))

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

        @property
        def indices(self):
            '''The indices corresponding to this superpixel's sub-pixels.'''
            return self._indices

    class Window(object):
        '''Represents a single 5x5 neighborhood in the image.'''
        def __init__(self, index, image):
            '''Instantiate a window at the given index of the given image.'''
            self._image = image
            self._index = index
            self._i = self._index[0]
            self._j = self._index[1]
            if self._image._color:
                self._window = self._image._labimage[self._i-2:self._i+3, self._j-2:self._j+3, :]
            else:
                self._window = self._image._raw[self._i-2:self._i+3, self._j-2:self._j+3]

            if image._color:
                lumin = np.median(self._window[...,0].astype(float))
                alpha = np.median(self._window[...,1].astype(float))
                beta = np.median(self._window[...,2].astype(float))
                self._median_color = (alpha, beta)
                self._variance = np.var(self._window[...,0].astype(float))
            else:
                lumin = np.median(self._window.astype(float))
                self._variance = np.var(self._window.astype(float))
            self._median_intensity = lumin

        @property
        def median_color(self):
            return self._median_color

        @property
        def median_intensity(self):
            return self._median_intensity

        @property
        def variance(self):
            return self._variance

        @property
        def window(self):
            return self._window

        @property
        def index(self):
            return (self._i - 2, self._j - 2)

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

    def _configure_neighborhoods(self):
        '''Configures the neighborhoods and their statistics for this image.'''
        # We need all of the neighborhoods for a grayscale image.
        self._windows = []
        step = 5 if self._color else 1
        for i in range(2, self._raw.shape[0] - 2, step):
            for j in range(2, self._raw.shape[1] - 2, step):
                self._windows.append(Image.Window((i, j), self))

    def _generate_variance_histogram(self):
        '''Configures the variance histogram for this image.'''
        sorted_windows = sorted(self._windows, key=lambda x: x.variance)
        self._variance_histogram = dict()
        if self._stat == 'variance-freq':
            self._variance_edges = np.zeros((50,), dtype=float)
            bin_size = len(sorted_windows) / 50

            for i in range(50):
                self._variance_histogram[i] = sorted_windows[:bin_size]
                sorted_windows = sorted_windows[bin_size:]
                self._variance_edges[i] = self._variance_histogram[i][-1].variance
            if len(sorted_windows) > 0:
                while len(sorted_windows) > 0:
                    self._variance_histogram[i].append(sorted_windows.pop(0))
                self._variance_edges[-1] = self._variance_histogram[i][-1].variance
        else:
            histo, self._variance_edges = np.histogram([w.variance for w in sorted_windows], bins=50)
            self._variance_edges = self._variance_edges[1:]

            for i in xrange(50):
                self._variance_histogram[i] = []
                for j in xrange(histo[i]):
                    self._variance_histogram[i].append(sorted_windows.pop(0))

    def __init__(self, path, color=False, stat='luminance'):
        '''Loads the image at the given path.'''
        self._path = path
        self._color = color
        self._stat = stat
        if self._color:
            self._raw = cv2.cvtColor(cv2.imread(path, cv.CV_LOAD_IMAGE_COLOR), cv.CV_BGR2RGB)
        else:
            self._raw = cv2.imread(path, cv.CV_LOAD_IMAGE_GRAYSCALE)

        if self._raw is None:
            raise BadImageError()

        if color:
            self._labimage = cv2.cvtColor(self._raw, cv.CV_RGB2Lab)

        if stat == 'luminance':
            # Calculate superpixels
            self._configure_superpixels()

            # Generate the grayscale histogram
            self._generate_grayscale_histogram()
        elif stat == 'variance-freq' or stat == 'variance-width':
            # Calculate the pixel neighborhoods.
            self._configure_neighborhoods()

            # Generate the variance histogram.
            if self._color:
                self._generate_variance_histogram()
        else:
            raise ValueError('Do not recognize the statistic: %s' % stat)

    def __iter__(self):
        '''You can iterate over the pixel groupings if you would like to.'''
        if self._stat == 'luminance':
            return iter(self._superpixels)
        elif self._stat == 'variance-freq' or self._stat == 'variance-width':
            return iter(self._windows)

    def __len__(self):
        '''The "length" of the image is represented by the number of superpixels in it.'''
        if self._stat == 'luminance':
            return len(self._superpixels)
        elif self._stat == 'variance-freq' or self._stat == 'variance-width':
            return len(self._windows)

    @property
    def raw(self):
        '''The raw RGB/L matrix returned by OpenCV (DO NOT MODIFY, UNDEFINED BEHAVIOR).'''
        return self._raw

    @property
    def shape(self):
        '''The shape as returned for the raw numpy representation.'''
        return self._raw.shape

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

    def lookup_variance(self, variance):
        '''Looks up all of the pixel windows that exist in the same variance bin as the provided
        variance value.
        '''
        indices = (self._variance_edges >= variance).nonzero()
        return self._variance_histogram[indices[0][0]] if len(indices[0]) > 0 else self._variance_histogram[49]

    def graph_grayscale_distribution(self, normalized=False):
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
