'''The final ML project of Kira Freijo and Amelia Reiss
Topic: Speed Dating!
Algorithms: Bayes Classifier and Decision Trees'''

import numpy as np
import random
import argparse
import pdb
import os.path
from pathlib import Path

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Train a perceptron to classify letters.")
parser.add_argument('-s', '--seed', help='seed to control randomness', type=int)

def main(args):

    # import data from csv file
    seed = random.seed(args.seed) # use seed parameter if given
    import os.path

    ROOT = os.path.dirname(os.path.abspath(__file__))
    rawdata = np.genfromtxt(ROOT + '/SDdata.csv', dtype=str, delimiter=",", filling_values="0")

    # randomize and separate data into training and testing
    rawdata = rawdata.reshape(7972, 65)
    labels = rawdata[0] # array of column headers
    data = np.delete(rawdata, 0, axis=0) # delete the first line of data array before randomizing
    
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

    # convert remaining data to floats
    useful_data[useful_data == ""] = "0"
    useful_data = useful_data.astype(np.float)

    # Bayes section
    xtrain = useful_data[0:7000, :]
    ytrain = data_labels[0:7000]
    xtest = useful_data[7000:, :]
    ytest = data_labels[7000:]

    num_training_documents = len(xtrain)
    num_testing_documents = len(xtest)
    # num_words = len(vocabulary)
    # num_newsgroups = len(newsgroups)
    num_attributes = len(attribute_labels)

    print("\n=======================")
    print("TRAINING")
    print("=======================")

    # Estimate the prior probabilities
    print("Estimating prior probabilities via MLE...")
    priors = get_priors(ytrain)

    # Estimate the class conditional probabilities
    print("Estimating class conditional probabilities via MAP...")
    
    # gotta change the class conditionals here to use the attributes (?)
    class_conditionals = get_class_conditionals(ytrain, xtrain, num_words, num_newsgroups)

    print("\n=======================")
    print("TESTING")
    print("=======================")

    # Test the Naive Bayes classifier
    print("Applying natural log to prevent underflow...")
    log_priors = np.log(priors)
    
    log_class_conditionals = np.log(class_conditionals)
    #print(log_class_conditionals[:,1])

    print("Counting words in each document...")
    counts = np.zeros((num_testing_documents, num_words))
    for doc in xtest:
        row = doc[0]
        col = doc[1]
        num = doc[2]
        counts[row][col] += num


    print("Computing posterior probabilities...")
    #access the words and multiply by the likelihood the word is in a class times all classes
    #then add all of the words together from the document

    # log_posterior = np.zeros((10, num_newsgroups))
    # for row in range(0, 10):
    #     log_posterior[row, :] = log_priors
    #     for word in range(num_words):
    #         log_posterior[row, :] += (counts[row, word] * log_class_conditionals[:, word])
            
    log_posterior = np.zeros((num_testing_documents, num_newsgroups))
    for row in range(num_testing_documents):
        log_posterior[row, :] = log_priors
        log_posterior[row, :] += np.sum(log_class_conditionals * counts[row,:], axis=1)


    print("Assigning predictions via argmax...")
    pred = np.argmax(log_posterior, axis=1)

    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compute performance metrics
    accuracy = (np.sum(pred == ytest) / len(pred)) * 100.0
    
    print(f"Accuracy: {np.sum(pred == ytest)}/{len(ytest)} ({round(accuracy, 2)}%)")
    
    print("Confusion matrix:")
    matrix = confusion_matrix(ytest, pred)
    
    #print confusion matrix
    for line in matrix:
        line2print = ""
        for number in line:
            spacing = " "*(5-len(str(number)))
            line2print += f"{spacing}{str(number)}"
        print(line2print)

    # Compare the accuracy
    score = classifier.score(xtest, ytest)
    print(f"The accuracy for this model is {score * 100}%")

    

def get_priors(labels):
    options, counts = np.unique(labels, return_counts=True)
    priors = np.zeros(len(options))
    for x in range(0, len(priors)):
        priors[x] = counts[x] / len(labels)

    return priors

def get_class_conditionals(labels, data, num_words, num_groups):
    cc = np.zeros((num_groups, num_words))
    for x in data:
        doc_id = x[0]
        row = labels[doc_id]
        col = x[1]
        num = x[2]
        #loads the count of each word in each label
        cc[row][col] += num
    
    #add the constant to everything to prevent problems
    cc += 1/num_words


    #sum all of the words in a doc type, then divide the specific word by total and save
    for y in range(0, len(cc)):
        row = cc[y]
        sum = np.sum(row)
        for word in range(0, len(cc[y])):
            cc[y][word] = cc[y][word] / sum

    return cc




if __name__ == "__main__":
    main(parser.parse_args())