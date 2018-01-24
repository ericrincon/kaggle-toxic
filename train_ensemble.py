"""
Generic training script will run most models that are passed in the command line
arguments from --model <model name>
"""

from toxic_text.train.util import build_base_arg_parser
from toxic_text.train.experiment import run_experiment
def main():
    argument_parser = build_base_arg_parser()
    argument_parser.add_argument('--models')
    args = argument_parser.parse_args()

    if args.models is None:
        models = ["sentence", 'dpcnn', 'birnn', 'clstm']
    else:
        models = args.models.split()

    for model_name in models:
        print('------------------------------------------------------'
              '----------------------------------------------')
        print('Training model: {}'.format(model_name))
        print('------------------------------------------------------'
              '----------------------------------------------')
        args.model = model_name
        run_experiment(args)


if __name__ == '__main__':
    main()