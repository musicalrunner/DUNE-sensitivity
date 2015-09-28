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
        # physics parameters
        self.params = Parameters.neutrinoParams_best.copy()
        # Number of neutrinos produced that will hit detector
        self.neutrino_energy = 3 * U.GeV
        self.baseline = 1300 * U.km
        self.params['deltaCP'] = math.pi / 2

        self.oscillator = Oscillator.Oscillator.fromParameterSet(self.params, U.rho_e,
                self.neutrino_energy)

    def calculateOscillation(self):
        """
        Calculate and save the oscillation probabilities.

        """
        initial_state = Oscillator.NeutrinoState(0, 1, 0, True);
        final_state = self.oscillator.evolve(initial_state,
                self.baseline);
        self.initial_state = initial_state
        self.final_state = final_state
