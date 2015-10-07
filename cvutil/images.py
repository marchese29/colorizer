import cv2
from cv2 import cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic

class BadImageError(Exception):
    pass

class NotColorError(AttributeError):
    pass

class Image(object):
    class Superpixel(object):
        def __init__(self, image, indices):
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

        @property
        def median_color(self):
            if self._image._color:
                return self._median_color
            else:
                raise NotColorError('Attempted color access of grayscale superpixel.')

        @property
        def median_intensity(self):
            return self._median_intensity

    def _configure_superpixels(self):
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
            self._superpixels.append(Image.Superpixel(self, indices))

    def __init__(self, path, color=False):
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

        self._configure_superpixels()

    def __iter__(self):
        return iter(self._superpixels)

    @property
    def raw(self):
        return self._raw

    def display(self, superpixels=False):
        fig = plt.figure('Image: (%d, %d)' % (self._raw.shape[0], self._raw.shape[1]))
        ax = fig.add_subplot(1, 1, 1)
        if superpixels:
            ax.imshow(mark_boundaries(self._raw, self._segments))
        else:
            ax.imshow(self._raw, cmap=plt.cm.gray, vmin=0, vmax=255)
        plt.axis('off')
        plt.show()
