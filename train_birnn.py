"""
Generic training script will run most models that are passed in the command line
arguments from --model <model name>
"""

from toxic_text.train.util import build_base_arg_parser
from toxic_text.train.experiment import run_experiment


def main():
    argument_parser = build_base_arg_parser()
    args = argument_parser.parse_args()

    run_experiment(args)


if __name__ == '__main__':
    main()