import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
Logistic Regression

'''

# Sigmoid function
def sigmoid(scores):
    return 1 / ( 1 + np.exp(-scores) )

# Log Likelihood Function
def log_likelihood( features, target, weights ):
    scores = np.dot( features, weights )
    LL = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return LL

# Main Logistic Regression Function
def logistic_regression( features, target, num_steps, learning_rate, add_intercept = False ):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        
        features = np.hstack((intercept, features))

    weights = np.zeros( features.shape[1] )

    # Iterating the loop again and again to Improve The Weights
    for step in range( num_steps ):

        # Dot Product Calculation
        scores = np.dot( features, weights )
        
        predictions = sigmoid( scores )

        # Update weights with log likelihood gradient
        output_error_signal = target - predictions

        gradient = np.dot( features.T, output_error_signal )
        
        weights = weights + learning_rate * gradient

        if step % 10000 == 0:
            print ( log_likelihood(features, target, weights) )
        
    return weights

# Importing the Dataset
train_data_set = pd.read_csv('Website Phishing.csv')
train_data_set.head()

a = len( train_data_set[train_data_set.Result==0] )
b = len( train_data_set[train_data_set.Result==-1] )
c = len( train_data_set[train_data_set.Result==1] )
print( a, "times 0 repeated in Result" )
print( b, "times -1 repeated in Result" )
print( c, "times 1 repeated in Result" )

train_data_set.describe()

# Training Data
train_data_set.info()

sns.countplot( train_data_set['Result'] )

sns.heatmap( train_data_set.corr(), annot=True )

sns.pairplot(train_data_set)

sns.heatmap( train_data_set.isnull(), cmap='Blues' )

# Separating X and Y from the input training data set
X = train_data_set.drop( 'Result',axis=1 ).values 
y = train_data_set['Result'].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split( X,y,test_size=0.25)

# Results of the split
print( "Training set has {} samples.".format( X_train.shape[0]) )
print( "Testing set has {} samples.".format( X_test.shape[0]) )

from sklearn.linear_model import LogisticRegression

#create logistic regression object
Classifier=LogisticRegression( random_state= 0, multi_class='multinomial' , solver='newton-cg' ) 
#Train the model using training data
Classifier.fit( X_train, y_train )


#import Evaluation metrics 
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
 
#Test the model using testing data
predictions = Classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix( y_test, predictions )
sns.heatmap( cm,annot=True )

print( "f1 score is ",f1_score(y_test, predictions, average='weighted') )

# Matthews Correlation Coefficient Produces a High Score only
# if the prediction is good for all of the confusion matrix categories
print( "Matthews correlation coefficient is ",matthews_corrcoef(y_test, predictions) )

print( "The accuracy of Logistic Regression Classifier on testing data is: ",100.0 *accuracy_score(y_test,predictions) )



# Seperating the Dataset 
input_data_set = pd.read_csv ('Website Phishing.csv') 

x = input_data_set.iloc[:, :-1]
y = input_data_set.iloc[:, : 1]
z = input_data_set.iloc[:, : 0]
x.head()

# Normalization is done to normalize the input data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xScaler = scaler.fit_transform(x)

# transform the given Labels to 0's and 1's
# Holdout Method to Split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xScaler,y, test_size = 0.25)


# Constructing The KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

k = 1
knn = KNeighborsClassifier ( n_neighbors=k )
knn.fit( x_train, y_train )

y_pred = knn.predict( x_test )
print("\nThe accuracy of KNN on testing data is:  ",metrics.accuracy_score( y_test, y_pred))


# Matthews Correlation Coefficient Produces a High Score only
# if the prediction is good for all of the confusion matrix categories
print("Matthews correlation coefficient is ",matthews_corrcoef(y_test, y_pred))


# Cross-validation and Evaluation
from sklearn.model_selection import cross_val_predict, cross_val_score

score = cross_val_score( knn, xScaler, y, cv = 8 )
#print( "\nCross-Validation Score :  ", score)

# Confusion Matrix Construction
y_prediction = cross_val_predict( knn, xScaler, y, cv = 10 )


confusion_matrix = metrics.confusion_matrix( y , y_prediction )
print( "\nConfusion Matrix :  \n",confusion_matrix)



# Calculating f1 Score
f1 = metrics.f1_score( y, y_prediction, average="weighted" )
print( "\nThe f1 Score is:  ",f1)


# Cross-Validation Accuracy Calculation
accuracy1 = metrics.accuracy_score(y, y_prediction)
print( "\nCross-Validation Accuracy Of KNN : ",accuracy1)
