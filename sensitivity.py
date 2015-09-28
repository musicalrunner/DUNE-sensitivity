"""
Calculate DUNE sensitivity to physics parameters.

Author: Sam Kohn

"""

import Oscillator.NeutrinoParameters as Parameters
import Oscillator.Units as U
import Oscillator.Oscillator as Oscillator
import math

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
        num_deltaCP_values = 100
        deltaCP_max = math.pi
        deltaCP_min = -math.pi
        self.param_sets = [Parameters.neutrinoParams_best.copy() for i
                in range(num_deltaCP_values)]
        for i in range(num_deltaCP_values):
            param_set = self.param_sets[i]
            param_set['deltaCP'] = (deltaCP_min + (deltaCP_max -
                    deltaCP_min)/num_deltaCP_values * i)

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
