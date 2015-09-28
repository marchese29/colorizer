import cv2
from cv2 import cv
from skimage.util import img_as_float
from skimage import io

class BadImageError(Exception):
    pass

class Image(object):
    def __init__(self, path, color=False):
        self._path = path
        self._color = color
        self._skimage = img_as_float(io.imread(self._path, as_grey=self._color))
        if self._color:
            self._cvimage = cv2.imread(path, cv.CV_LOAD_IMAGE_COLOR)
        else:
            self._cvimage = cv2.imread(path, cv.CV_LOAD_IMAGE_GRAYSCALE)
        if self._cvimage is None:
            raise BadImageError()

    @property
    def skimage(self):
        return self._skimage

    @property
    def cvimage(self):
        return self._cvimage

    @property
    def color(self):
        return self._color
    
    
    