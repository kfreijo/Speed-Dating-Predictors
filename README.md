# Speed Dating!

What kind of attributes influence a couple's compatibility? Here, we utilize decision trees and Bayes' classifiers to predict whether or not a couple will match up at the end of a night of speed dating. 

# About the Data
The data we used was found on Kaggle, and it was collected from multiple [Speed Dating Experiments](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment?resource=download) from 2002-2004. The file has over 7000 entries, providing details like age, gender, and how much they like to go to the movies. 

# Required Libraries
* numpy
* scikit-learn
* matplotlib

# Files Included
***SDdata.csv***
This csv file contains the data imported from Kaggle, used in our algorithms. Note: entries with more than 6 values left blank were not included in our analysis. 

***ogSpeedDatingData.csv***
The original dataset with all of the columns and rows that the original researchers collected.

***Speed Dating Data Key.doc***
A word document containing a key for the meaning behind the variables in the dataset.

***SpeedDatingTree.py***
This python file creates and plots a decision tree that determines whether or not someone will match, based on the attribute values.

The confusion matrix and accuracy for the model is printed in the terminal, and the decision tree is visualized with only the first 7 rows from the classifier, for easier consumption. Additionally, after closing the tree plot, a bar graph with the values from the confusion matrix will appear, (blue bars meaning correctly classified outcomes and red bars meaning incorrectly classified outcomes).


***SpeedDatingNB.py***
This python file utilizes a Naive Bayes Classifier to predict if someone will match or not, based on the values of the given attributes. 

The confusion matrix and accuracy for the model is printed in the terminal, and the confusion matrix is plotted on a bar graph, (blue bars meaning correctly classified outcomes and red bars meaning incorrectly classified outcomes).