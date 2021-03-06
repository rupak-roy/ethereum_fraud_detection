#etherum fraud detection
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
dataset = pd.read_csv("transaction_dataset.csv",sep=",")
dataset = dataset.iloc[:,2:]
dataset.columns = dataset.columns.str.replace(' ', '')
#checking if we can group
from collections import Counter
len(Counter(dataset.Address))
def counts (data):
    counts = Counter(data)
    print(counts)
    print("length",len(counts))
counts(dataset.Address)
counts(dataset.ERC20_most_rec_token_type)
counts(dataset.ERC20mostsenttokentype)
d =dataset.groupby(dataset.ERC20mostsenttokentype).mean()
X = d.iloc[:,1:]
y =  d["FLAG"] 
y = np.round(y).astype(int)
y.groupby(y).size()
#Class imbalance issue
X.isna().any()
def outlier_dec(data):
    sns.boxplot(data)
    
outlier_dec(X.Avgminbetweensenttnx)
#----------------------------------------------------------
# Create correlation matrix
corr_matrix = X.corr().abs()
#------Remove the highly correlated variables----------------------
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]
# Drop features 
X.drop(to_drop, axis=1, inplace=True)
#Drop columns that have 1 unique value -------------------------
X.loc[:,X.nunique()!=1]
X.drop(columns=X.columns[X.nunique()==1], inplace=True)
X = X.values
#----------------------------
def evaluation_score (y_test,y_pred):
    cm = confusion_matrix(y_test,y_pred) 
    print("Confusion Matrix \n", cm)
    print('Balanced Accuracy ',metrics.balanced_accuracy_score(y_test,y_pred))
    print("Recall Accuracy Score~TP",metrics.recall_score(y_test, y_pred))
    print("Precision Score ~ Ratio of TP",metrics.precision_score(y_test, y_pred))
    print("F1 Score",metrics.f1_score(y_test, y_pred))
    print("auc_roc score", metrics.roc_auc_score(y_test,y_pred))
    print("Classification Report", classification_report(y_test,y_pred))
    
    
def cross_validation(model,X_train,y_train,n):
    kfold = KFold(n_splits=10)  
    accuracies = cross_val_score(model,X= X_train,y= y_train,cv = kfold,scoring='accuracy')
    print("Standard Deviation",accuracies.std())
    print("Mean/Avergae Score",accuracies.mean())
#----------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators= 150,max_depth=100)
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
#Accuracy SCore
evaluation_score(y_test, y_pred)
cross_validation(rf_model,X_train,y_train,10)
#----------- WITH SMOTE
y 
X
# describes info about train and test set 
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape)
#----------------------------------------------
from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))
# import SMOTE module from imblearn library 
# pip install imblearn (if you don't have imblearn in your system) 
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2,sampling_strategy=1) 
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))
def train_model(n,max_d):
    rf_model = RandomForestClassifier(n_estimators=n,max_depth=max_d)
    rf_model.fit(X_train_res,y_train_res.ravel())
    predictions = rf_model.predict(X_test)
    return predictions
def show_predictions(data):
    results = rf_model.predict(data)
    return results 
    
    
train_model_predictions = train_model(500,100)
#Accuracy Score--------------------------------------
evaluation_score(y_test, train_model_predictions)
cross_validation(rf_model,X_train,y_train,10)
#Select the best model------------------------------------------
def get_models():
    models = dict()
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    models['svm'] = SVC()
    models['random_forest'] = RandomForestClassifier()
    models['bayes'] = GaussianNB()
    return models
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
 return scores
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
 scores = evaluate_model(model, X, y)
 results.append(scores)
 names.append(name)
 print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()

#GridSeachCV -----
from sklearn.model_selection import GridSearchCV
p = [{'n_estimators':[50,100,150],'max_depth':[10, 100]}]
'''p = [{'n_estimators':[50,100,150],'max_depth':[10, 100],
               'min_samples_split':[2,3,4,5,6,7,8,9,10],'min_samples_leaf':[2,3,4,5],
               'min_impurity_decrease':[2,3,4,5],'max_features':["auto","sqrt","log2"]}]'''
grid_search = GridSearchCV(estimator = rf_model,param_grid= p, scoring = "accuracy",cv=10,n_jobs=-1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("best accuracy , best_paramters", best_accuracy, best_parameters)
#-------------------------------------------
train_model_predictions1= train_model(100,10)
#Accuracy Score
evaluation_score(y_test, train_model_predictions1)
cross_validation(rf_model,X_train,y_train,10)
show_predictions(X_test)
