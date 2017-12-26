import argparse
import pandas as pd
import numpy as np

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--data')
    argument_parser.add_argument('--output')

    args = argument_parser.parse_args()

    data =