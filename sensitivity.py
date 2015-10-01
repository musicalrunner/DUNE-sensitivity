"""
Calculate DUNE sensitivity to physics parameters.

Author: Sam Kohn

"""

import Oscillator.NeutrinoParameters as Parameters
import Oscillator.Units as U
import Oscillator.Oscillator as Oscillator
import operator
import numpy as np
import matplotlib.pyplot as plt
import random

class SensitivityCalculator(object):
    """
    Calculate the sensitivity to various physics parameters.

    """
    
    def __init__(self, energyInGeV=3):
        """
        Create a new SensitivityCalculator for muon neutrino
        disappearance.

        """
        self.neutrino_energy = energyInGeV * U.GeV
        self.baseline = 1300 * U.km
        # Delta CP values to calculate oscillation parameters for
        self.num_deltaCP_values = 100
        self.deltaCP_max = np.pi
        self.deltaCP_min = -np.pi
        self.syst_errors = [0.02, 0.05, 0.02, 0.05]
        self.chi_squares = []
        # Parameter sets representing the different values of delta CP
        self.param_sets = [Parameters.neutrinoParams_best.copy() for i
                in range(self.num_deltaCP_values)]
        for i in range(self.num_deltaCP_values):
            param_set = self.param_sets[i]
            param_set['deltaCP'] = (self.deltaCP_min + (self.deltaCP_max -
                    self.deltaCP_min)/self.num_deltaCP_values * i)

        # Create an oscillation calculator for each value of delta CP
        self.oscillators = [Oscillator.Oscillator.fromParameterSet(params, U.rho_e,
                self.neutrino_energy) for params in self.param_sets]

    def calculateOscillations(self):
        """
        Calculate and save the oscillation probabilities.

        """
        nu_initial_state = Oscillator.NeutrinoState(0, 1, 0, True);
        nu_final_states = [oscillator.evolve(nu_initial_state,
            self.baseline) for oscillator in self.oscillators]

        nubar_initial_state = Oscillator.NeutrinoState(0, 1, 0, False);
        nubar_final_states = [oscillator.evolve(nubar_initial_state,
            self.baseline) for oscillator in self.oscillators]

        self.nu_initial_state = nu_initial_state
        self.nubar_initial_state = nubar_initial_state
        self.nu_final_states = nu_final_states
        self.nubar_final_states = nubar_final_states
        return (nu_final_states, nubar_final_states)

    def plotProbabilities(self):
        """
        Plot the probabilities of observing a given flavor for the range
        of delta CP values given in the constructor.

        """

        if not hasattr(self, 'nu_final_states'):
            self.calculateOscillations()
        x_values = np.arange(self.deltaCP_min, self.deltaCP_max,
                (self.deltaCP_max -
                    self.deltaCP_min)/self.num_deltaCP_values)
        y_values = [[state.probabilities()[flavor] for flavor in [0,1]]
                for state in self.nu_final_states]
        plt.plot(x_values, y_values)
        y_values = [[state.probabilities()[flavor] for flavor in [0,1]]
                for state in self.nubar_final_states]
        plt.plot(x_values, y_values)
        plt.legend(self.legendString())
        plt.show()

    def _chiSquares(self, num_detecteds, num_produceds):
        """
        Return the chi-square for each value of delta CP for the given
        detected number of e and mu neutrinos, assuming there were num_produced
        neutrinos before oscillations, with the given uncorrelated
        relative uncertainties.

        """
        if not hasattr(self, 'nu_final_states'):
            self.calculateOscillations()

        # Take the first two neutrino flavors' probabilities multiplied
        # by the number of (anti)neutrinos produced
        nu_num_expecteds = \
                [np.asarray(state.probabilities()[0:2])*num_produceds[0]
                for state in self.nu_final_states]
        nubar_num_expecteds = \
                [np.asarray(state.probabilities()[0:2])*num_produceds[1]
                for state in self.nubar_final_states]

        # Compute sigma by multiplyint the relative uncertainty by the
        # number of detected (anti)neutrinos
        nu_sigmas = map(operator.mul,
                self.syst_errors[:2], num_detecteds[:2])
        nubar_sigmas = map(operator.mul,
                self.syst_errors[2:], num_detecteds[2:])

        chiSquares = []
        for num_expected_by_flavor in zip(nu_num_expecteds,
                nubar_num_expecteds):
            chiSquare = 0
            for neutrino_flavor in [0,1]:
                # first neutrinos, then antineutrinos
                expected = num_expected_by_flavor[0][neutrino_flavor]
                detected = num_detecteds[:2][neutrino_flavor]
                sigma = nu_sigmas[neutrino_flavor]
                # error^2 = (sigma (syst.))^2 + (sqrt(N))^2
                chiSquare += ((expected - detected)**2/(sigma*sigma +
                    detected))
                # now antineutrinos
                expected = num_expected_by_flavor[1][neutrino_flavor]
                detected = num_detecteds[2:][neutrino_flavor]
                sigma = nubar_sigmas[neutrino_flavor]
                chiSquare += ((expected - detected)**2/(sigma*sigma +
                    detected))
            chiSquares.append(chiSquare)

        self.chi_squares = chiSquares
        return chiSquares

    def testDeltaCPs(self):
        """
        Return an array of the values of delta CP used as test values.

        """
        step = (self.deltaCP_max -
                self.deltaCP_min)/self.num_deltaCP_values
        return [self.deltaCP_min + step * i for i in range(self.num_deltaCP_values)]

    def detectedEvents(self, nu_num_produced, nubar_num_produced,
            deltaCP, includeErrors=True):
        """
        Get the number of events that would be detected in a perfect
        detector given the number of unoscillated neutrinos and deltaCP.

        """
        if not hasattr(self, 'nu_initial_state'):
            self.calculateOscillations()
        params = Parameters.neutrinoParams_best.copy()
        params['deltaCP'] = deltaCP
        newOscillator = Oscillator.Oscillator.fromParameterSet(params,
                U.rho_e, self.neutrino_energy)
        nu_detected = np.asarray(newOscillator.evolve(self.nu_initial_state,
                self.baseline).probabilities()) * nu_num_produced
        nubar_detected = np.asarray(newOscillator.evolve(self.nubar_initial_state,
                self.baseline).probabilities()) * nubar_num_produced
        num_detecteds = np.concatenate((nu_detected[:2], nubar_detected[:2]))
        if includeErrors:
            # Statistical errors
            num_detecteds = map(lambda n: n + random.gauss(0, np.sqrt(n)),
                    num_detecteds)
            # Systematic errors
            num_detecteds = [n + random.gauss(0, err*n) for (n, err) in
                    zip(num_detecteds, self.syst_errors)]
        return num_detecteds

    @staticmethod
    def legendString():
        """
        Returns a list of strings to use as the legend.

          - nu e
          - nu mu
          - nu e bar
          - nu mu bar
        """
        return [r"$\nu_{e}$", r"$\nu_{\mu}$", r"$\bar{\nu}_{e}$",
                r"$\bar{\nu}_{\mu}$"]

    def chiSquares(self, num_neutrinos, true_deltaCP):
        """
        Calculate the chi square for different values of delta CP under the
        given experimental conditions:

          - num_neutrinos: the number of neutrinos before oscillation.
            The effective number of antineutrinos is calculated as a
            fraction of the number of neutrinos.
          - syst_errors: a list of relative systematic errors in the
            order [nu_e, nu_mu, nu_e_bar, nu_mu_bar]
          - true_deltaCP: the value of deltaCP to use to calculate
            oscillations.

        """
        random.seed(0)
        self.chi_squares = []
        num_antineutrinos = 0.1 * num_neutrinos
        num_detecteds = self.detectedEvents(num_neutrinos,
                num_antineutrinos, true_deltaCP)
        num_produceds = [num_neutrinos, num_antineutrinos]
        self.chi_squares = self._chiSquares(num_detecteds, num_produceds)
        return self.chi_squares

    def plotChiSquare(self, num_neutrinos, true_deltaCP):
        """
        Plot the chi square for different values of delta CP under the
        given experimental conditions:

          - num_neutrinos: the number of neutrinos before oscillation.
            The effective number of antineutrinos is calculated as a
            fraction of the number of neutrinos.
          - syst_errors: a list of relative systematic errors in the
            order [nu_e, nu_mu, nu_e_bar, nu_mu_bar]
          - true_deltaCP: the value of deltaCP to use to calculate
            oscillations.

        """
        figure = plt.figure()
        deltaCP_values = [delta/np.pi for delta in self.testDeltaCPs()]
        chiSquares = self.chiSquares(num_neutrinos, true_deltaCP)
        minChiSquare = min(chiSquares)
        deltaChiSquares = [chiSquare - minChiSquare for chiSquare in
                chiSquares]
        rootChiSquare_values = [np.sqrt(deltaChiSquare) for
                deltaChiSquare in deltaChiSquares]
        axes = figure.add_subplot(111)
        axes.plot(deltaCP_values, rootChiSquare_values)
        axes.set_xticks(deltaCP_values[::10])
        axes.set_xlabel(r"$\delta_{CP}/\pi$")
        axes.set_ylabel(r"$\sqrt{\Delta\chi^{2}}$")
        axes.set_xlim(deltaCP_values[0], deltaCP_values[-1])
        return figure
