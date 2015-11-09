'''Contains utilities for representing the various distributions needed in this project.'''
from matplotlib import collections as mc
from matplotlib import pyplot as plt

from util import max_index

class LabColorDistribution(object):
    '''Represents the two-dimensional color distribution using the alpha and beta channels of the
    Lab color space as axes.
    '''
    class ColorBin(object):
        '''Represents a bin containing several superpixels.

        bounds: [(a_min, a_max), (b_min, b_max)]
        '''
        def __init__(self, superpixels, originating_split, dist, bounds=[(0, 255), (0, 255)]):
            '''Generates a color bin from the given superpixel.'''
            for sp in superpixels:
                sp._color_bin = self
            self._dim = (originating_split + 1) % 2
            self._superpixels = sorted(superpixels, key=lambda x: x.median_color[self._dim])
            self._dist = dist
            self._bounds = bounds

        def __len__(self):
            return len(self._superpixels)

        @property
        def index(self):
            if not hasattr(self, '_idx'):
                raise AttributeError('No index has been set on this color bin yet.')
            return self._idx

        def set_index(self, idx):
            self._idx = idx

        def split(self):
            '''Splits the color bin along the opposite axis from the previous split.'''
            split_pixel = self._superpixels[len(self._superpixels) / 2]
            if self._dim == 0:
                self._dist._lines.append(
                    [(split_pixel.median_color[0], self._bounds[1][0]),
                    (split_pixel.median_color[0], self._bounds[1][1])]
                )
                bounds1 = [
                    (self._bounds[0][0], split_pixel.median_color[0]),  # Alphas cut maximum
                    (self._bounds[1][0], self._bounds[1][1])  # Betas are the same
                ]
                bounds2 = [
                    (split_pixel.median_color[0], self._bounds[0][1]),  # Alphas cut minimum
                    (self._bounds[1][0], self._bounds[1][1])  # Betas are the same
                ]
            else:
                self._dist._lines.append(
                    [(self._bounds[0][0], split_pixel.median_color[1]),
                    (self._bounds[0][1], split_pixel.median_color[1])]
                )
                bounds1 = [
                    (self._bounds[0][0], self._bounds[0][1]),  # Alphas are the same
                    (self._bounds[1][0], split_pixel.median_color[1])  # Betas cut maximum
                ]
                bounds2 = [
                    (self._bounds[0][0], self._bounds[0][1]),  # Alphas are the same
                    (split_pixel.median_color[1], self._bounds[1][1])  # Betas cut minimum
                ]

            bin1 = LabColorDistribution.ColorBin(self._superpixels[:len(self._superpixels)/2],
                self._dim, self._dist, bounds=bounds1)
            bin2 = LabColorDistribution.ColorBin(self._superpixels[len(self._superpixels)/2:],
                self._dim, self._dist, bounds=bounds2)
            return (bin1, bin2)

        @property
        def average_color(self):
            '''The average color in this bin.'''
            # Check for a cached value.
            if hasattr(self, '_rep_color'):
                return self._rep_color

            # Function for accumulating superpixel colors into the total point.
            def accum(point, sp):
                t1 = point[0] + sp.median_color[0]
                t2 = point[1] + sp.median_color[1]
                return (t1, t2)

            # Calculate the representative totals
            totals = reduce(lambda tot, sp: accum(tot, sp), self._superpixels, initializer=(0, 0))
            self._rep_color = (totals[0] / len(self._superpixels), totals[1] / len(self._superpixels))

    def _configure_bins(self, first_bin, num_bins):
        '''Configures the color bins to have the desired number of bins using the provided initial
        all-encompassing bin.
        '''
        self._bins = [first_bin]

        while len(self._bins) < num_bins:
            # Find the largest bin.
            biggest_idx = max_index(self._bins, key=len)
            biggest_bin = self._bins.pop(biggest_idx)

            # Split the largest bin at the median.
            (bin1, bin2) = biggest_bin.split()

            # Add the splits back to the list of bins.
            self._bins.append(bin1)
            self._bins.append(bin2)

    def __init__(self, context_images, num_bins=75):
        '''Builds the color distribution from the provided context images using the requested number
        of bins.
        '''
        self._num_bins = num_bins
        self._superpixels = []
        for image in context_images:
            self._superpixels += [sp for sp in image]

        # Actually generate the distribution here.
        init_bin = LabColorDistribution.ColorBin(self._superpixels, 1, self)
        self._lines = []
        self._configure_bins(init_bin, num_bins)

    def __iter__(self):
        return iter(self._bins)

    def lookup(self, superpixel):
        '''Looks up the given superpixel in this distribution and returns the corresponding
        (alpha, beta) tuple.
        '''
        return superpixel._color_bin

    def display(self, bins=False):
        '''Display all of the points in this distribution, also shows the bin boundaries if bins is
        True.
        '''
        alphas = [sp.median_color[0] for sp in self._superpixels]
        betas = [sp.median_color[1] for sp in self._superpixels]
        ax = plt.axes()
        ax.scatter(alphas, betas, marker='.')
        if bins:
            lines = mc.LineCollection(self._lines, linewidths=2, colors=[(1, 0, 0, 1)])
            ax.add_collection(lines)
        plt.show()

    @property
    def bins(self):
        '''The list of color bins.'''
        return self._bins

    @property
    def max_distance(self):
        '''Returns the maximum euclidean distance between any two color bins.'''
        if hasattr(self, '_max_distance'):
            return self._max_distance

        self._max_distance = 0.0
        for i in range(len(self._bins)-1):
            for j in range(i+1, len(self._bins)):
                distance = LabColorDistribution.bin_distance(self._bins[i], self._bins[j])
                if distance > self._max_distance:
                    self._max_distance = distacne
        return self._max_distance

    @staticmethod
    def bin_distance(bin1, bin2):
        '''Determines the Lab color distance between the average values of the given bin.'''
        return sum(map(lambda x, y: (x - y)**2.0, bin1.average_color, bin2.average_color))**0.5
