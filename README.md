# Speed Dating!

What kind of attributes influence a couple's compatibility? Here, we utilize decision trees and Bayes' classifiers to predict whether or not a couple will match up at the end of a night of speed dating. 

# About the Data
The data we used was found on Kaggle, and it was collected from multiple [Speed Dating Experiments](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment?resource=download) from 2002-2004. The file has over 7000 entries, providing details like age, gender, and how much they like to go to the movies. 

# Required Libraries
* numpy
* scikit-learn
* matplotlib

# Files Included
**SDdata.csv**
This csv file contains the data imported from Kaggle, used in our algorithms. Note: entries with more than 6 values left blank were not included in our analysis. 

**SpeedDatingTree.py**
This python file creates and plots a decision tree that determines whether or not a pair will be a match, based on the attribute values. 

**SpeedDatingBayes.py**
This python file utilizes a Naive Bayes Classifier to predict if a pair will be a match or not, based on the values of the given attributes. 