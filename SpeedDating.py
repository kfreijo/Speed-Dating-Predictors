'''The final ML project of Kira Freijo and Amelia Reiss
Topic: Speed Dating!
Algorithms: Bayes Classifier and Decision Trees'''

import numpy as np
import random
import argparse
import pdb

parser = argparse.ArgumentParser(description="Train a perceptron to classify letters.")
parser.add_argument('-s', '--seed', help='seed to control randomness', type=int)

def main(args):

    # import data from csv file
    random.seed(args.seed) # use seed parameter if given
    data = np.genfromtxt("SDdata.csv", dtype=str, delimiter=",", filling_values="x")

    # randomize and separate data into training and testing
    labels = data[0] # array of column headers
    data = np.delete(data, 0) # delete the first line of data array before randomizing


if __name__ == "__main__":
    main(parser.parse_args())