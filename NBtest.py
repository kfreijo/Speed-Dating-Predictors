'''The final ML project of Kira Freijo and Amelia Reiss
Topic: Speed Dating!
Algorithms: Bayes Classifier and Decision Trees'''

import numpy as np
import random
import argparse
import pdb
import os.path
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Train a perceptron to classify letters.")
parser.add_argument('-s', '--seed', help='seed to control randomness', type=int)

def main(args):

    # import data from csv file
    seed = random.seed(args.seed) # use seed parameter if given
    ROOT = os.path.dirname(os.path.abspath(__file__))
    with open("SDdata.csv") as file:
        labels = file.readline().split(",")
    data = np.genfromtxt(ROOT + '/SDdata.csv', dtype=int, delimiter=",", filling_values="0", skip_header=1)

    # randomize and separate data into training and testing
    data = data.reshape(7971, 65)
    
    # extract the id numbers in case we need
    # delete the id number cols
    p1id = data[:, [0,1]]
    p2id = data[:, 6]
    data_labels = data[:, 7]
    useful_data = np.delete(data, 0, axis=1)
    useful_data = np.delete(useful_data, 0, axis=1)
    useful_data = np.delete(useful_data, 4, axis=1)
    useful_data = np.delete(useful_data, 4, axis=1)

    # delete the same info from the labels so they match the data
    attribute_labels = np.delete(labels, 0)
    attribute_labels = np.delete(attribute_labels, 0)
    attribute_labels = np.delete(attribute_labels, 4)
    attribute_labels = np.delete(attribute_labels, 4)

    # delete correlation column
    useful_data = np.delete(useful_data, 4, axis=1)
    useful_data[useful_data == ""] = 0

    # Bayes section
    xtrain = useful_data[0:7000, :]
    ytrain = data_labels[0:7000].astype(int)
    xtest = useful_data[7000:, :]
    ytest = data_labels[7000:]


    # train model
    clf = GaussianNB()
    clf.fit(xtrain, ytrain)

    # predict results
    ypred = clf.predict(xtest)

    cm = confusion_matrix(ytest, ypred)
    print(cm)
    #accuracy score of the model
    print('Accuracy score :', accuracy_score(ytest, ypred))

    plt.figure(figsize=(10,5))
    plt.title("Confusion matrix")

if __name__ == "__main__":
    main(parser.parse_args())