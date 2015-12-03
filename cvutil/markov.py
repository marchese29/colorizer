'''Represents the markov random field that we will eventually solve for the coloring.'''
import numpy as np
from maxflow import fastmin

from cvutil.distributions import LabColorDistribution

class MarkovGraph(object):
    '''Represents the graph generated for solving the Markov Random Field.'''
    def _configure_smoothness_term(self):
        '''This configures the matrix representing the penalty for picking two different color
        labels at neighboring pixels.
        '''
        self._V = np.zeros((self._D.shape[-1],)*2, dtype=float)
        self._distribution.max_distance
        for i in range(self._V.shape[0]):
            bin1 = self._distribution.bins[i]
            for j in range(self._V.shape[1]):
                bin2 = self._distribution.bins[j]
                dist = LabColorDistribution.bin_distance(bin1, bin2) / self._distribution.max_distance
                self._V[i,j] = dist
                self._V[j,i] = dist

    def _configure_variation_likelihoods(self):
        '''This configures the matrix representing the likelihood of a color variation along each
        axis.
        '''
        self._U = np.zeros((self._D.shape[0], self._D.shape[1], 2), dtype=float)

        # Get the raw target image as a floating point array.
        raw = self._target.astype(float)

        # Find the absolute difference between the neighbors along each dimension
        self._U[:,:-1,1] = np.absolute(raw[:,:-1] - raw[:,1:])
        self._U[:-1,:,0] = np.absolute(raw[:-1,:] - raw[1:,:])

        # Normalize everything.
        self._U /= self._U.max()
        self._U = 1.0 - self._U

        # Invalid neighbors are set to infinity
        self._U[:,-1,0] = np.inf
        self._U[-1,:,1] = np.inf

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

        # Configure the variation likelihoods.
        # self._configure_variation_likelihoods()

    def solve(self):
        '''Solves the MRF using the Fast Approximate Energy Minimization technique.  Returns the
        labelling in the form of a numpy array.
        '''
        if hasattr(self, '_solution'):
            return self._solution
        self._solution = fastmin.aexpansion_grid(self._D, self._V)
        return self._solution
