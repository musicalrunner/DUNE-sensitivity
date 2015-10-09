from sensitivity import *
import argparse
import string

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
    args = parser.parse_args()
    spec = spectrum[args.spectrum]
    params = args.parameter
    outfilename = args.outFileName
    numValues = args.numValues
    ordering = args.ordering
    form = args.form

    outfilenames = outfilename
    if len(params) > 1:
        nameParts = string.split(outfilename, '.')
        name = string.join(nameParts[:-1], '.')
        extension = '.' + nameParts[-1]
        outfilenames = [name + '_' + param + extension for param in params]
    else:
        outfilenames = [outfilename]

    for outfilename, param in zip(outfilenames, params):
        print 'Generating plots for parameter "' + param + '"'
        fig = plot2dDetectionMaps(spec, param, ordering, numValues)
        fig.savefig(outfilename, format=form)
        del fig

if __name__ == "__main__":
    main()
