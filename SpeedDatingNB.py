'''The final ML project of Kira Freijo and Amelia Reiss
Topic: Speed Dating!
Algorithms: Bayes Classifier and Decision Trees'''

import numpy as np
import pdb
import os.path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

def main():

    # import data from csv file
    ROOT = os.path.dirname(os.path.abspath(__file__))
    with open(ROOT + "/SDdata.csv") as file:
        labels = file.readline().split(",")
    data = np.genfromtxt(ROOT + '/SDdata.csv', dtype=int, delimiter=",", filling_values="0", skip_header=1)

    # reshape data and extract ID numbers
    data = data.reshape(7971, 65)
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

    # delete correlation column since data was imported as ints
    useful_data = np.delete(useful_data, 4, axis=1)
    useful_data[useful_data == ""] = 0

    # separate data into training and testing
    xtrain = useful_data[0:7000, :]
    ytrain = data_labels[0:7000].astype(int)
    xtest = useful_data[7000:, :]
    ytest = data_labels[7000:]

    # train model
    clf = GaussianNB()
    clf.fit(xtrain, ytrain)

    # predict results
    ypred = clf.predict(xtest)

    # create confusion matrix for predicted results
    cm = confusion_matrix(ytest, ypred)
    print("Confusion Matrix:")
    for line in cm:
        line2print = ""
        for number in line:
            spacing = " "*(5-len(str(number)))
            line2print += f"{spacing}{str(number)}"
        print(line2print)
    
    # print accuracy score of the model
    print('Accuracy score :', accuracy_score(ytest, ypred) * 100)

    # create and print bar graph of confusion matrix
    plt.bar(["Correct non-matches", "False non-matches", "Correct matches", "False matches"], [cm[0][0], cm[1][0], cm[1][1], cm[0][1]], color=["blue", "red", "blue", "red"])
    plt.title("Naive Bayes Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()