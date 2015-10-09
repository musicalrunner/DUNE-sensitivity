from sensitivity import *
import argparse
import string
import json

def main():
    parser = argparse.ArgumentParser(description='Draw 2D Maps of ' +
            'detected neutrinos.')
    parser.add_argument('parameter', type=str, nargs='+',
            help='Name(s) of parameters to compare with delta CP')
    parser.add_argument('-n', help='Number of parameter values to test',
            default=10, type=int, dest='numValues')
    parser.add_argument('-o', help='Output file name', type=str,
            dest='outFileName', required=True)
    parser.add_argument('-s', '--spectrum', help='Name of spectrum',
            type=str, choices=spectrum.keys(), required=True)
    parser.add_argument('-O', '--ordering', choices=["NO", "IO"],
            required=True, help='The mass hierarchy/ordering to use')
    parser.add_argument('-f', '--format', required=False, default=None,
            help='The image format to use', dest='form')
    parser.add_argument('-q', '--quiet', action='store_true',
            help='Suppress printing output')
    parser.add_argument('-d', '--save-data', action='store_true',
            dest='save', help='Save the numerical data in ' +
            'output/<outFileNamePrefix>.json')
    args = parser.parse_args()
    spec = spectrum[args.spectrum]
    params = args.parameter
    outfilename = args.outFileName
    numValues = args.numValues
    ordering = args.ordering
    form = args.form
    quiet = args.quiet
    save = args.save

    outfilenames = outfilename
    nameParts = string.split(outfilename, '.')
    name = string.join(nameParts[:-1], '.')
    extension = '.' + nameParts[-1]
    datafilename = 'output/' + name + '.json'
    if len(params) > 1:
        outfilenames = ['output/' + name + '_' + param + extension for param in params]
    else:
        outfilenames = ['output/' + outfilename]

    data = {}
    for outfilename, param in zip(outfilenames, params):
        qprint('Generating plots for parameter "' + param + '"', quiet)
        fig, nues, numus, nuebars, numubars = plot2dDetectionMaps(spec, param, ordering, numValues)
        if save:
            try:
                data[param] = {
                        "nue":  np.asarray(nues).tolist(),
                        "numu": np.asarray(numus).tolist(),
                        "nuebar":  np.asarray(nuebars).tolist(),
                        "numubar":  np.asarray(numubars).tolist(),
                        }
            except:
                qprint('Could not convert data to python list.', quiet)
                data[param] = {
                        "nue": nues,
                        "numu": numus,
                        "nuebar": nuebars,
                        "numubar": numubars
                        }
        fig.savefig(outfilename, format=form)
        del fig
    if save:
        with open(datafilename, 'w') as f:
            qprint('Dumping JSON data to ' + f.name, quiet)
            try:
                json.dump(data, f)
            except TypeError, e:
                qprint('Could not serialize data to JSON: ' + str(e),
                        quiet)

def qprint(output, quiet):
    if not quiet:
        print output

if __name__ == "__main__":
    main()
