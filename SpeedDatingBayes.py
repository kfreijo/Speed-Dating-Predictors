'''The final ML project of Kira Freijo and Amelia Reiss
Topic: Speed Dating!
Algorithms: Bayes Classifier and Decision Trees'''

import numpy as np
import random
import argparse
import pdb
import os.path
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import math

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

    num_training_pairs = len(xtrain)
    num_testing_pairs = len(xtest)
    num_attributes = len(attribute_labels)

    print("\n=======================")
    print("TRAINING")
    print("=======================")

    # Estimate the prior probabilities
    print("Estimating prior probabilities via MLE...")
    priors = np.bincount(ytrain) / 2

    # Estimate the class conditional probabilities
    print("Estimating class conditional probabilities via MAP...")
    # class conditionals is number of times each attribute has certain value for match(y/n)
    # initialize 2 arrays that will hold the counts of each value type 
    class_conditionals1 = np.zeros((num_attributes-1, 101)).astype(int)
    class_conditionals2 = np.zeros((num_attributes-1, 101)).astype(int)

    nomatch = xtrain[ytrain==0, :] # rows that were not a match
    matches = xtrain[ytrain==1, :] # rows that were a match

    for attribute in range(0, num_attributes-1):
        pdb.set_trace()
        class_conditionals1[attribute] = np.sum(nomatch[:, attribute], axis=1) 
        class_conditionals2[attribute] = np.sum(matches[:, attribute], axis=1)

    class_conditionals = np.zeros((num_attributes-1, 2)).astype(int)
    class_conditionals = np.zeros((101, num_attributes))

    pdb.set_trace()

    for attribute in range(0, len(class_conditionals)):
        class_conditionals[attribute, 0] += np.sum(class_conditionals1[attribute])
        class_conditionals[attribute, 1] += np.sum(class_conditionals2[attribute])

    class_conditionals = class_conditionals.transpose().astype(float)

    pdb.set_trace()
    # class_conditionals /= np.average(class_conditionals, 0)


    # rows = xtrain[:, 1].tolist()
    # cols = ytrain[xtrain[:, 0]].tolist()
    # np.add.at(class_conditionals, (rows, cols), xtrain[:, 2])
    # alpha = (1 / num_attributes)
    # class_conditionals += alpha


    print("\n=======================")
    print("TESTING")
    print("=======================")

    # Test the Naive Bayes classifier
    print("Applying natural log to prevent underflow...")
    log_priors = np.log(priors)
    log_class_conditionals = np.log(class_conditionals)
    log_class_conditionals[log_class_conditionals == -math.inf] = 0
    #print(log_class_conditionals[:,1])

    print("Calculating attributes from each pair...")
    counts = np.zeros((num_attributes, 101)).astype(int)
    # counts = counts.transpose()


    for attribute in range(0, len(xtest[0])):
        temp_count = np.bincount(xtest[:, attribute])
        for value in counts[attribute]:
            counts[attribute, value] = temp_count[value]


    print("Computing posterior probabilities...")
    pdb.set_trace()
    log_posterior = np.matmul(counts, log_class_conditionals)
    for match in ytest:
        if match == 0:
            log_posterior[match]
    log_posterior += log_priors
            
    # log_posterior = np.zeros((num_testing_pairs, 2))
    # for row in range(num_testing_pairs):
    #     log_posterior[row, :] = log_priors
    #     log_posterior[row, :] += np.sum(log_class_conditionals * counts[row,:], axis=1)



    print("Assigning predictions via argmax...")
    pred = np.argmax(log_posterior, 1)
    pdb.set_trace()

    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compute performance metrics
    accuracy = (np.sum(pred == ytest) / len(pred)) * 100.0
    print(f"Accuracy: {np.sum(pred == ytest)}/{len(ytest)} ({round(accuracy, 2)}%)")
    # matrix = confusion_matrix(ytest, pred)
    # print("Confusion matrix:")
    
    # # print confusion matrix
    # for line in matrix:
    #     line2print = ""
    #     for number in line:
    #         spacing = " "*(5-len(str(number)))
    #         line2print += f"{spacing}{str(number)}"
    #     print(line2print)

    # Compare the accuracy
    # score = classifier.score(xtest, ytest)
    score = np.mean(ytest == pred)
    pdb.set_trace()
    print(f"The accuracy for this model is {score * 100}%")


def get_class_conditionals(labels, data, num_attributes):
    cc = np.zeros((2, num_attributes))
    for x in data:
        doc_id = x[0]
        row = labels[doc_id]
        col = x[1]
        num = x[2]
        #loads the count of each word in each label
        cc[row][col] += num
    
    #add the constant to everything to prevent problems
    cc += 1/num_attributes

    #sum all of the words in a doc type, then divide the specific word by total and save
    for y in range(0, len(cc)):
        row = cc[y]
        sum = np.sum(row)
        for word in range(0, len(cc[y])):
            cc[y][word] = cc[y][word] / sum

    return cc




if __name__ == "__main__":
    main(parser.parse_args())