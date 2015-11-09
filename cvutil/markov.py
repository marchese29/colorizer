'''Represents the markov random field that we will eventually solve for the coloring.'''
import numpy as np
from maxflow import fastmin

class MarkovGraph(object):
    '''Represents the graph generated for solving the Markov Random Field.'''
    def _configure_smoothness_term(self):
        '''This configures the matrix representing the penalty for picking two different color
        labels at neighboring pixels.
        '''
        self._V = np.zeros((self._D.shape[-1],)*2, dtype=float)
        for i in range(self._V.shape[0]):
            bin1 = self._distribution.bins[i]
            for j in range(self._V.shape[1]):
                bin2 = self._distribution.bins[j]
                dist = LabColorDistribution.bin_distance(bin1, bin2) / self._distribution.max_distance
                self._V[i,j] = dist
                self._V[j,i] = dist

    def __init__(self, target_image, lab_distribution, probability_distribution, smoothness=1.0):
        '''Configure the markov random field using the given target and reference images, using the
        provided number of labels for the coloring.
        '''
        self._target = target_image
        self._distribution = lab_distribution
        self._labels = range(probability_distribution.shape[-1])
        self._D = 1.0 - probability_distribution

        # Configure the smoothness term.
        self._smoothness = smoothness
        self._configure_smoothness_term()
        self._V *= smoothness

    def solve(self):
        '''Solves the MRF using the Fast Approximate Energy Minimization technique.  Returns the
        labelling in the form of a numpy array.
        '''
        if hasattr(self, '_solution'):
            return self._solution
        self._solution = fastmin.aexpansion(self._D, self._V)
        return self._solution
