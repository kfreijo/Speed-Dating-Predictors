'''The final ML project of Kira Freijo and Amelia Reiss
Topic: Speed Dating!
Algorithms: Bayes Classifier and Decision Trees'''

import numpy as np
import random
import argparse
import pdb
import os.path
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def main():

    # import data from csv file
    import os.path
    ROOT = os.path.dirname(os.path.abspath(__file__))
    rawdata = np.genfromtxt(ROOT + '/SDdata.csv', dtype=str, delimiter=",", filling_values="0")

    # reshape data and get labels
    rawdata = rawdata.reshape(7972, 65)
    labels = rawdata[0] # array of column headers
    data = np.delete(rawdata, 0, axis=0) # delete the first line of data array before randomizing
    
    # extract the id numbers and delete from data
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

    # convert remaining data to floats
    useful_data[useful_data == ""] = "0"
    useful_data = useful_data.astype(np.float)

    # separate data into training and testing
    xtrain = useful_data[0:7200, :]
    ytrain = data_labels[0:7200]
    xtest = useful_data[7200:, :]
    ytest = data_labels[7200:]

    # create decision tree based on entropy
    classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 7)
    classifier.fit(xtrain, ytrain)

    # Test the decision tree
    prediction = classifier.predict(xtest)

    # Show the confusion matrix for test data
    matrix = confusion_matrix(ytest, prediction)
    
    # print the confusion matrix
    print("Confusion Matrix:")
    for line in matrix:
        line2print = ""
        for number in line:
            spacing = " "*(5-len(str(number)))
            line2print += f"{spacing}{str(number)}"
        print(line2print)

    # Compare the accuracy
    score = classifier.score(xtest, ytest)
    print(f"The accuracy for this model is {score * 100}%")

    # Visualize the tree using matplotlib and plot_tree
    plt.figure()
    labels = np.array(["Did not match", "Matched"])
    tree = plot_tree(classifier, 
                feature_names = attribute_labels,
                class_names = labels,
                rounded = True,
                filled = True,
                fontsize=7,
                max_depth=6)
    plt.show()

    # print the bar graph of the confusion matrix
    plt.bar(["Correct non-matches", "False non-matches", "Correct matches", "False matches"], [matrix[0][0], matrix[1][0], matrix[1][1], matrix[0][1]], color=["blue", "red", "blue", "red"])
    plt.title("Decision Tree Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()