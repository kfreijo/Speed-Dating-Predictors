'''The final ML project of Kira Freijo and Amelia Reiss
Topic: Speed Dating!
Algorithms: Bayes Classifier and Decision Trees'''

import numpy as np
import pdb

def main():
    # import data from csv file
    data = np.loadtxt("SpeedDatingData.csv", dtype=str, delimiter=",")
    labels = data[0]
    print(labels)


if __name__ == "__main__":
    main()