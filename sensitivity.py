"""
Calculate DUNE sensitivity to physics parameters.

Author: Sam Kohn

"""

import Oscillator.NeutrinoParameters as Parameters
import Oscillator.Units as U
import Oscillator.Oscillator as Oscillator
import operator
import numpy as np

class SensitivityCalculator(object):
    """
    Calculate the sensitivity to various physics parameters.

    """
    
    def __init__(self):
        """
        Create a new SensitivityCalculator for muon neutrino
        disappearance.

        """
        self.neutrino_energy = 3 * U.GeV
        self.baseline = 1300 * U.km
        # physics parameters
        self.num_deltaCP_values = 100
        self.deltaCP_max = np.pi
        self.deltaCP_min = -np.pi
        self.param_sets = [Parameters.neutrinoParams_best.copy() for i
                in range(self.num_deltaCP_values)]
        for i in range(self.num_deltaCP_values):
            param_set = self.param_sets[i]
            param_set['deltaCP'] = (self.deltaCP_min + (self.deltaCP_max -
                    self.deltaCP_min)/self.num_deltaCP_values * i)

        self.oscillators = [Oscillator.Oscillator.fromParameterSet(params, U.rho_e,
                self.neutrino_energy) for params in self.param_sets]

    def calculateOscillations(self):
        """
        Calculate and save the oscillation probabilities.

        """
        initial_state = Oscillator.NeutrinoState(0, 1, 0, True);
        final_states = [oscillator.evolve(initial_state, self.baseline)
                for oscillator in self.oscillators]
        self.initial_state = initial_state
        self.final_states = final_states
        return final_states

    def plotProbabilities(self, flavor=1):
        """
        Plot the probabilities of observing a given flavor for the range
        of delta CP values given in the constructor.

        """
        import matplotlib.pyplot as plt

        if not hasattr(self, 'final_states'):
            self.calculateOscillations()
        x_values = np.arange(self.deltaCP_min, self.deltaCP_max,
                (self.deltaCP_max -
                    self.deltaCP_min)/self.num_deltaCP_values)
        y_values = [state.probabilities()[flavor] for state in
                self.final_states]
        plt.plot(x_values, y_values)
        plt.show()

    def chiSquares(self, num_detecteds, num_produced,
            relative_uncertainties):
        """
        Return the chi-square for each value of delta CP for the given
        detected number of e and mu neutrinos, assuming there were num_produced
        neutrinos before oscillations, with the given uncorrelated
        relative uncertainties.

        """
        if not hasattr(self, 'final_states'):
            self.calculateOscillations()

        num_expecteds = [np.asarray(state.probabilities()[:-1])*num_produced for state in
                self.final_states]

        sigmas = map(operator.mul, relative_uncertainties, num_detecteds)
        chiSquares = []
        for num_expected_by_flavor in num_expecteds:
            chiSquare = 0
            for neutrino_type in [0,1]:
                expected = num_expected_by_flavor[neutrino_type]
                detected = num_detecteds[neutrino_type]
                sigma = sigmas[neutrino_type]
                chiSquare += ((expected - detected)/sigma)**2
            chiSquares.append(chiSquare)

        return chiSquares

    def testDeltaCPs(self):
        """
        Return an array of the values of delta CP used as test values.

        """
        step = (self.deltaCP_max -
                self.deltaCP_min)/self.num_deltaCP_values
        return [self.deltaCP_min + step * i for i in range(self.num_deltaCP_values)]

    def detectedEvents(self, num_produced, deltaCP):
        """
        Get the number of events that would be detected in a perfect
        detector given the number of unoscillated neutrinos and deltaCP.

        """
        if not hasattr(self, 'initial_state'):
            self.calculateOscillations()
        params = Parameters.neutrinoParams_best.copy()
        params['deltaCP'] = deltaCP
        newOscillator = Oscillator.Oscillator.fromParameterSet(params,
                U.rho_e, self.neutrino_energy)
        return np.asarray(newOscillator.evolve(self.initial_state,
                self.baseline).probabilities()) * num_produced
