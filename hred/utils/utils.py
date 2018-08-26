import argparse

def get_args(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--n_hidden', type=int, default= 128)

    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    

    return parser.parse_args(argv)

