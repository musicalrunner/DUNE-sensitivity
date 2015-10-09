"""
Calculate DUNE sensitivity to physics parameters.

Author: Sam Kohn

"""

import Oscillator.NeutrinoParameters as Parameters
import Oscillator.Units as U
import Oscillator.Oscillator as Osc
from Oscillator.Oscillator import NeutrinoState
import operator
import numpy as np
import matplotlib.pyplot as plt

class SensitivityCalculator(object):
    """
    Calculate the sensitivity to various physics parameters.

    """
    
    def __init__(self, deltaCP_max, deltaCP_min, num_deltaCPs, spectrum,
            oscParameters, GLoBES):
        """
        Create a new SensitivityCalculator for muon neutrino
        disappearance.

        The spectrum must be a sequence of (energyInGeV, relativeAmount)
        pairs. The neutrino sample will then be composed of a sum of
        discrete energy samples. The relative amounts do not have to add
        to 1.

        """
        self.spectrum = spectrum
        self.params = oscParameters
        self.energies, self.energyWeights = map(list, zip(*spectrum))
        self.energies = [E * U.GeV for E in self.energies]
        self.energyWeights = (np.asarray(self.energyWeights,
            dtype=np.float64) / sum(self.energyWeights))
        self.baseline = 1300 * U.km
        # Delta CP values to calculate oscillation parameters for
        self.num_deltaCP_values = num_deltaCPs
        self.deltaCP_max = deltaCP_max
        self.deltaCP_min = deltaCP_min
        self.syst_errors = [0.02, 0.05, 0.02, 0.05]
        self.chi_squares = []
        # Parameter sets representing the different values of delta CP
        if not oscParameters:
            oscParameters = Parameters.neutrinoParams_best
        self.param_sets = [oscParameters.copy() for i
                in range(self.num_deltaCP_values)]
        for i in range(self.num_deltaCP_values):
            param_set = self.param_sets[i]
            param_set['deltaCP'] = (self.deltaCP_min + (self.deltaCP_max -
                    self.deltaCP_min)/(self.num_deltaCP_values-1) * i)

        # Create an oscillation calculator for each value of delta CP
        self.globes = GLoBES
        if GLoBES:
            import Oscillator.Oscillator_GLoBES
            self.oscillatorType = Oscillator.Oscillator_GLoBES.Oscillator_GLoBES
            self.oscillators = [[
                (self.oscillatorType.fromParameterSet, (params, U.rho_e,
                neutrino_energy)) for neutrino_energy in self.energies] for
                params in self.param_sets]
        else:
            self.oscillatorType = Osc.Oscillator
            self.oscillators = [[
                self.oscillatorType.fromParameterSet(params, U.rho_e,
                float(neutrino_energy)) for neutrino_energy in self.energies] for
                params in self.param_sets]

    @classmethod
    def sensitivityTester(cls, spectrum, oscParameters=None, GLoBES=False):
        """
        Return a SensitivityCalculator set up to scan over fine-grained
        delta-CP values.

        """
        if not hasattr(spectrum, "__len__"):
            spectrum = [(spectrum, 1)]
        return cls(np.pi, -np.pi, 100, spectrum, oscParameters, GLoBES)

    @classmethod
    def probabilityViewer(cls, spectrum, oscParameters=None, GLoBES=False):
        """
        Return a SensitivityCalculator set up over a few delta-CP
        values.

        """
        if not hasattr(spectrum, "__len__"):
            spectrum = [(spectrum, 1)]
        return cls(np.pi/2, -np.pi/2, 3, spectrum, oscParameters, GLoBES)

    def calculateOscillations(self):
        """
        Calculate and save the oscillation probabilities.

        """
        nu_initial_state = NeutrinoState(0, 1, 0, True);
        # for oscillatorList in self.oscillators:
            # for (oscillator, params) in oscillatorList:
                # o = apply(oscillator, params)
                # state = o.evolve(nu_initial_state,
                        # self.baseline)
                # print state.probabilities()
                # del o
        if self.globes:
            nu_final_states = [[apply(oscillator, params).evolve(nu_initial_state,
                self.baseline) for oscillator, params in oscSubList] for oscSubList
                in self.oscillators]
        else:
            nu_final_states = [[oscillator.evolve(nu_initial_state,
                self.baseline) for oscillator in oscSubList] for oscSubList
                in self.oscillators]

        nubar_initial_state = NeutrinoState(0, 1, 0, False);
        if self.globes:
            nubar_final_states = [[apply(oscillator,
                params).evolve(nubar_initial_state,
                self.baseline) for oscillator, params in oscSubList] for oscSubList
                in self.oscillators]
        else:
            nubar_final_states = [[oscillator.evolve(nubar_initial_state,
                self.baseline) for oscillator in oscSubList] for oscSubList
                in self.oscillators]

        self.nu_initial_state = nu_initial_state
        self.nubar_initial_state = nubar_initial_state
        self.nu_final_states = nu_final_states
        self.nubar_final_states = nubar_final_states
        return (nu_final_states, nubar_final_states)

    def getNumberOfObservedNeutrinos(self, numNu=1, numNubar=1):
        """
        Get the number of observed neutrinos and antineutrinos at a
        range of delta CP values. The given neutrino numbers are the
        number of unoscillated neutrinos and antineutrinos.

        Output:
            A 5-tuple containing lists of corresponding delta-CP values,
            neutrino (e, mu) detections and antineutrino (e, mu) detections.
        """
        if not hasattr(self, 'nu_final_states'):
            self.calculateOscillations()

        # Combine the results of the oscillations into a single counting
        # measurement over all the energies in the spectrum.
        detections = []
        for spectrum in self.nu_final_states:
            e, mu = 0, 0
            for energyResult, weight in zip(spectrum, self.energyWeights):
                probabilities = energyResult.probabilities()
                e  += probabilities[0] * weight
                mu += probabilities[1] * weight
            detections.append((e, mu))
        electrons, muons = map(np.asarray, zip(*detections))
        electrons *= numNu
        muons *= numNu
        nubar_detections = []
        for spectrum in self.nubar_final_states:
            ebar, mubar = 0, 0
            for energyResult, weight in zip(spectrum, self.energyWeights):
                probabilities = energyResult.probabilities()
                ebar += probabilities[0] * weight
                mubar += probabilities[1] * weight
            nubar_detections.append((ebar, mubar))
        nubar_electrons, nubar_muons = map(np.asarray,
                zip(*nubar_detections))
        nubar_electrons *= numNubar
        nubar_muons *= numNubar
        return (self.testDeltaCPs(), electrons, muons, nubar_electrons,
                nubar_muons)

    def plotProbabilities(self, energyBin=0, showPlot=True):
        """
        Plot the probabilities of observing a given flavor for the range
        of delta CP values given in the constructor.

        """
        if not hasattr(self, 'nu_final_states'):
            self.calculateOscillations()
        x_values = self.testDeltaCPs()
        y_values = [[state.probabilities()[flavor] for flavor in [0,1]]
                for state in zip(*self.nu_final_states)[energyBin]]
        nu_values = y_values
        plt.plot(x_values, y_values)
        y_values = [[state.probabilities()[flavor] for flavor in [0,1]]
                for state in zip(*self.nubar_final_states)[energyBin]]
        nubar_values = y_values
        plt.plot(x_values, y_values)
        plt.xlabel(r"$\delta_{CP}$")
        plt.ylabel("Oscillation Probability at 1300 km")
        plt.title("Oscillation probability for " +
                str(self.energies[energyBin]/U.GeV) + " GeV neutrinos")
        plt.legend(self.legendString())
        if showPlot:
            plt.show()
        return (x_values, nu_values, nubar_values)

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
        # Sum the expected number over the energy spectrum
        # Note: zip(*list) has the effect of transposing the list
        nu_num_expecteds_by_energy = \
                [[np.asarray(state.probabilities()[0:2]) *
                    num_produceds[0] * weight for state, weight in
                    zip(energyStates, self.energyWeights)] for energyStates
                    in self.nu_final_states]
        nu_num_expecteds = [[sum(allEnergies) for allEnergies in
                zip(*allFlavors)] for allFlavors in
                nu_num_expecteds_by_energy]
        nubar_num_expecteds_by_energy = \
                [[np.asarray(state.probabilities()[0:2]) *
                    num_produceds[1] * weight for state, weight
                    in zip(energyStates, self.energyWeights)] for
                    energyStates in self.nubar_final_states]
        nubar_num_expecteds = [[sum(allEnergies) for allEnergies in
                zip(*allFlavors)] for allFlavors in
                nubar_num_expecteds_by_energy]

        # Compute sigma by multiplying the relative uncertainty by the
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
                self.deltaCP_min)/(self.num_deltaCP_values-1)
        return [self.deltaCP_min + step * i for i in range(self.num_deltaCP_values)]

    def detectedEvents(self, nu_num_produced, nubar_num_produced, deltaCP):
        """
        Get the number of events that would be detected in a perfect
        detector given the number of unoscillated neutrinos and deltaCP.

        """
        if not hasattr(self, 'nu_initial_state'):
            self.calculateOscillations()
        params = Parameters.neutrinoParams_best.copy()
        params['deltaCP'] = deltaCP
        newOscillators = [self.oscillatorType.fromParameterSet(params,
                U.rho_e, energy) for energy in self.energies]
        nu_detected_by_energy = [np.asarray(osc.evolve(self.nu_initial_state,
                self.baseline).probabilities()) * nu_num_produced
                * weight for osc, weight in zip(newOscillators,
                    self.energyWeights)]
        nu_detected = [sum(energies) for energies in
                zip(*nu_detected_by_energy)]
        nubar_detected_by_energy = [np.asarray(osc.evolve(self.nubar_initial_state,
                self.baseline).probabilities()) * nubar_num_produced
                * weight for osc, weight in zip(newOscillators,
                    self.energyWeights)]
        nubar_detected = [sum(energies) for energies in
                zip(*nubar_detected_by_energy)]
        return np.concatenate((nu_detected[:2], nubar_detected[:2]))

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
        axes.set_xlabel(r"$\delta_{CP}/\pi$")
        axes.set_ylabel(r"$\sqrt{\Delta\chi^{2}}$")
        axes.set_xlim(deltaCP_values[0], deltaCP_values[-1])
        return figure

# Some functions
def plotNuEAppearanceProb(antiNeutrino=False):
    """
    Plot the appearance probability for electron (anti)neutrinos as a
    function of energy at 1300km for deltaCP = +/- Pi/2 and 0. This is
    supposed to reproduce Figure 3.1 in the DUNE CDR.

    """
    # Calculate for a set of 100 energies ranging logarithmically from
    # 0.1 GeV to 10 GeV
    energies = [0.1 * 1.047129**n for n in range(100)]
    calcs = [SensitivityCalculator.probabilityViewer([(energy, 1)]) for energy in energies]
    [calc.calculateOscillations() for calc in calcs]
    nuStates = [zip(*calc.nu_final_states)[0] for calc in calcs]
    nue_probs = [[state.probabilities()[0] for state in cpState] for cpState
            in nuStates]
    figure = plt.figure()
    plt.grid(True)
    plt.ylim([0, 0.2])
    axes = figure.add_subplot(111)
    axes.set_color_cycle(['b', 'r', 'g'])
    axes.set_xscale('log')
    axes.set_xlabel("Neutrino Energy [GeV]")
    axes.set_ylabel("Oscillation Probability")
    axes.set_title("Oscillation Probability for Neutrinos over 1300 km")
    axes.plot(energies, nue_probs)
    [axes.fill_between(energies, probs, color=color) for probs, color in
            zip(zip(*nue_probs), ['b', 'r', 'g'])]
    legendStrings = [r"$\delta_{CP} = -\pi/2$", r"$\delta_{CP} = 0$",
            r"$\delta_{CP} = +\pi/2$"]
    plt.legend(legendStrings)
    return figure

def plotGLoBESvsDan(energy, antiNeutrino=False):
    """
    Plot the oscillation probabilities for muon and electron neutrinos
    for the GLoBES oscillator and for Dan's oscillator.

    WARNING: This method constructs multiple GLoBES
    objects, which will result in a segmentation violation when Python
    exits and tries to clean up the same global C++ object multiple
    times. In an interactive session, you must hit Ctrl-C to exit after
    the segmentation violation occurs.

    """
    message = """WARNING: This method constructs multiple GLoBES
    objects, which will result in a segmentation violation when Python
    exits and tries to clean up the same global C++ object multiple
    times. In an interactive session, you must hit Ctrl-C to exit after
    the segmentation violation occurs."""
    print message

    dan = SensitivityCalculator.sensitivityTester(energy, False)
    globes = SensitivityCalculator.sensitivityTester(energy, True)

    dan_values = dan.plotProbabilities(showPlot=False)
    globes_values = globes.plotProbabilities(showPlot=False)
    figure = plt.figure()
    axes = figure.add_subplot(111)
    antiText = ""
    if antiNeutrino:
        antiText = "anti"
    axes.set_xlabel(r"$\delta_{CP}$")
    axes.set_ylabel("Oscillation probability")
    axes.set_title("Oscillation Probability for " + str(energy) +
            " GeV " + antiText + "neutrinos over 1300km")
    neutrinoIndex = int(antiNeutrino) + 1
    axes.plot(dan_values[0], zip(*dan_values[neutrinoIndex])[0], 'r',
            dan_values[0], zip(*dan_values[neutrinoIndex])[1], 'k',
            globes_values[0], zip(*globes_values[neutrinoIndex])[0], 'b',
            globes_values[0], zip(*globes_values[neutrinoIndex])[1], 'm')
    legendStrings = ["Dan $e$", r"Dan $\mu$", "GLoBES $e$",
            r"GLoBES $\mu$"]
    plt.legend(legendStrings)
    return figure

def plot2dDetectionMaps(spectrum, parameter, hierarchy, numValues=10):
    """
    Plot the number of expected neutrinos as a function of delta CP and
    the given parameter. delta CP ranges from -pi to pi, and the given
    parameter ranges over +/- 3sigma on the latest (2014) NuFIT results.
    All other parameters are fixed at the NuFIT best fit results for the
    given hierarchy. hierarchy can be either "IO" or "NO" for inverted
    and normal, respectively.

    """
    parameterSet = Parameters.nufit_NO  # inverted ordering
    if hierarchy == "IO":  # inverted ordering
        parameterSet = Parameters.nufit_IO
    upper = parameterSet['+3sigma'][parameter]
    lower = parameterSet['-3sigma'][parameter]
    valuesToTest = np.linspace(upper, lower, numValues)
    param_set = [parameterSet['best'].copy() for _ in
            valuesToTest]
    for params, value in zip(param_set, valuesToTest):
        params[parameter] = value

    oscillators = [SensitivityCalculator.sensitivityTester(spectrum,
        oscParameters=param) for param in param_set]
    numNu = 100000
    numNubar = numNu/10.0
    valueSets = [osc.getNumberOfObservedNeutrinos(numNu, numNubar) for
            osc in oscillators]
    nues = [values[1] for values in valueSets]
    numus = [values[2] for values in valueSets]
    nuebars = [values[3] for values in valueSets]
    numubars = [values[4] for values in valueSets]
    figure = plt.figure(figsize=(11, 9))
    def imshow(ax, matrix):
        return ax.imshow(matrix, plt.get_cmap('spectral'), interpolation='nearest',
            extent=[-np.pi, np.pi, min(valuesToTest),
                max(valuesToTest)], aspect='auto')
    nue_axes = figure.add_subplot(2, 2, 1)
    numu_axes = figure.add_subplot(2, 2, 2)
    nuebar_axes = figure.add_subplot(2, 2, 3)
    numubar_axes = figure.add_subplot(2, 2, 4)

    nue_image = imshow(nue_axes, nues)
    numu_image = imshow(numu_axes, numus)
    nuebar_image = imshow(nuebar_axes, nuebars)
    numubar_image = imshow(numubar_axes, numubars)
    nue_axes.set_title(r"$\nu_{e}$ Appearance")
    numu_axes.set_title(r"$\nu_{\mu}$ Disappearance")
    nuebar_axes.set_title(r"$\bar{\nu}_{e}$ Appearance")
    numubar_axes.set_title(r"$\bar{\nu}_{\mu}$ Disappearance")

    nuebar_axes.set_xlabel(r"$\delta_{CP}$", fontsize=20)
    numubar_axes.set_xlabel(r"$\delta_{CP}$", fontsize=20)
    nue_axes.set_ylabel(parameter, fontsize=20)
    nuebar_axes.set_ylabel(parameter, fontsize=20)

    figure.colorbar(nue_image, ax=nue_axes)
    figure.colorbar(numu_image, ax=numu_axes)
    figure.colorbar(nuebar_image, ax=nuebar_axes)
    figure.colorbar(numubar_image, ax=numubar_axes)
    return figure, nues, numus, nuebars, numubars

spectrum = {
        'mono3': [(3, 1)],
        'mono2': [(2, 1)],
        'flat': zip(np.arange(1, 5, 1), [1]*4),
        'peak3': zip(np.arange(0.5, 4.5, 0.5), [.5, 1, 2, 3, 5, 8, 4,
            1]),
        }
