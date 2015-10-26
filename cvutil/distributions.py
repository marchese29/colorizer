'''Contains utilities for representing the various distributions needed in this project.'''
from util import max_index

class _ColorBin(object):
    '''Represents a bin containing several superpixels.'''
    def __init__(self, superpixels, originating_split):
        '''Generates a color bin from the given superpixel.'''
        [sp._color_bin = self for sp in superpixels]
        self._dim = (originating_split + 1) % 2
        self._superpixels = sorted(superpixels, key=lambda x: x.median_color[self._dim])

    def __len__(self):
        return len(self._superpixels)

    def __contains__(self, sp):
        return sp in self._superpixels

    def split(self):
        '''Splits the color bin along the opposite axis from the previous split.'''
        bin1 = _ColorBin(self._superpixels[:len(self._superpixels)], self._dim)
        bin2 = _ColorBin(self._superpixels[len(self._superpixels):], self._dim)
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

class LabColorDistribution(object):
    '''Represents the two-dimensional color distribution using the alpha and beta channels of the
    Lab color space as axes.
    '''
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
        init_bin = _ColorBin(self._superpixels, 1)
        self._configure_bins(init_bin, num_bins)

    def lookup(self, superpixel):
        '''Looks up the given superpixel in this distribution and returns the corresponding
        (alpha, beta) tuple.
        '''
        return superpixel._color_bin.average_color
