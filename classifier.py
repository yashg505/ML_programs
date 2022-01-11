
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_validate, ShuffleSplit
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

os.chdir('C:\\Users\\yashg\\Documents\\UCC\\Sem-2\\CS6405\\assignment\\Project-2')
df = pd.read_csv('dataset-sat.csv')
df.head(5)
df.drop('INSTANCE_ID', axis = 'columns', inplace = True)

x = df.drop('ebglucose_solved', axis = 'columns')
y = df['ebglucose_solved']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.7, random_state = 5)

#Feature normalization
def fnorm(x_train, x_test):
    norm = MinMaxScaler().fit(x_train)

    x_train_norm = pd.DataFrame(norm.transform(x_train))
    x_train_norm.columns = x_train.columns
    x_test_norm = pd.DataFrame(norm.transform(x_test))
    x_test_norm.columns = x_test.columns
    
    return x_train_norm, x_test_norm

#Feature Selection

def select_best_features(x_train, x_test, n):
    #n - number of best features
    # only works after normalization
    bestfeatures = SelectKBest(score_func=chi2, k=40)
    fit = bestfeatures.fit(x_train,y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_train.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    
    #print(featureScores.nlargest(40,'Score'))
    
    x = featureScores.nlargest(n,'Score').iloc[:,0]
    x_train_new = x_train.loc[:,x]
    x_test_new = x_test.loc[:,x]

    return x_train_new, x_test_new

#cannot use correlation matrix as there are too many features to propoerly visualize the data
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(400, 400))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)


#Metric definition to compare models
MLA_columns = ['MLA Name', 'MLA Accuracy' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)
MLA_predict = y.copy()

#KNN

def KNN_1(x_train, y_train, x_test, y_test):
    MLA_compare.loc[0, 'MLA Name'] = "KNN- Data Normalized, running on all features"
    model = KNeighborsClassifier()
    start = time.time()
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_test)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[0, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[0, 'MLA Time'] = end - start

def KNN_2(x_train, y_train, x_test, y_test):
    MLA_compare.loc[1, 'MLA Name'] = "KNN- Tuned"
    model = KNeighborsClassifier()
    start = time.time()
    params = [{'metric' : ['minkowski','euclidean','manhattan'],
               'weights' : ['uniform','distance'],
               'n_neighbors'  : np.arange(5,15)}]
    gs_knn = GridSearchCV(estimator = KNeighborsClassifier(n_jobs=-1),
                          param_grid=params,
                          scoring='accuracy',
                          cv=5)
    gs_knn.fit(x_train, y_train)
    train_predictions = gs_knn.predict(x_test)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[1, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[1, 'MLA Time'] = end - start
    
#Tree Based Algorithms
def rf_1(x_train, y_train, x_test, y_test):
    MLA_compare.loc[2, 'MLA Name'] = "RF, running on all features"
    model = RandomForestClassifier()
    start = time.time()
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_test)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[2, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[2, 'MLA Time'] = end - start

    return model

def rf_2(x_train, y_train, x_test, y_test):
    MLA_compare.loc[3, 'MLA Name'] = "RF, Tuned and running on selected features"
    model = RandomForestClassifier()
    start = time.time()
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
    rfecv.fit(x_train, y_train)
    n = rfecv.n_features_
    print("RF - RFECV")
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    feat_importances = pd.Series(rfecv.ranking_, index=x_train.columns)
    x = feat_importances.nsmallest(n).index
    x_train_new = x_train.loc[:,x]
    x_test_new = x_test.loc[:,x] 
    
    params = [{'n_estimators': [10, 50, 100, 300, 500],
               'criterion': ['gini', 'entropy'],
               'max_depth': [2, 4, 6, 8, 10, 15, 20, None],
               'oob_score': [True]
              }]
    gs_rf = GridSearchCV(estimator = RandomForestClassifier(n_jobs=-1),
                          param_grid=params,
                          scoring='accuracy',
                          cv=5)
    gs_rf.fit(x_train_new, y_train)
    train_predictions = gs_rf.predict(x_test_new)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[3, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[3, 'MLA Time'] = end - start

def gbc_1(x_train, y_train, x_test, y_test):
    MLA_compare.loc[4, 'MLA Name'] = "GradientBoosting Classifier, running on all features"
    model = GradientBoostingClassifier()
    start = time.time()
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_test)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[4, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[4, 'MLA Time'] = end - start

def gbc_2(x_train, y_train, x_test, y_test):
    MLA_compare.loc[5, 'MLA Name'] = "GradientBoosting Classifier, Tuned and running on selected features"
    model = GradientBoostingClassifier()
    start = time.time()
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
    rfecv.fit(x_train, y_train)
    n = rfecv.n_features_
    print("Gradient Boosting - RFECV")
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    feat_importances = pd.Series(rfecv.ranking_, index=x_train.columns)
    x = feat_importances.nsmallest(n).index
    x_train_new = x_train.loc[:,x]
    x_test_new = x_test.loc[:,x] 
    
    params = [{'loss': ['deviance', 'exponential'], 
               'learning_rate': [.05],
               'n_estimators': [300],
               'criterion': ['friedman_mse', 'mse', 'mae'],
               'max_depth': [2, 4, 6, 8, 10, None],
              }]
    gs_gbc = GridSearchCV(estimator = GradientBoostingClassifier(),
                          param_grid=params,
                          scoring='accuracy',
                          cv=5)
    gs_gbc.fit(x_train_new, y_train)
    train_predictions = gs_gbc.predict(x_test_new)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[5, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[5, 'MLA Time'] = end - start

def adab_1(x_train, y_train, x_test, y_test):
    MLA_compare.loc[6, 'MLA Name'] = "AdaBoostClassifier, running on all features"
    model = AdaBoostClassifier()
    start = time.time()
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_test)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[6, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[6, 'MLA Time'] = end - start

def adab_2(x_train, y_train, x_test, y_test):
    MLA_compare.loc[7, 'MLA Name'] = "AdaBoostClassifier, Tuned and running on selected features"
    model = AdaBoostClassifier()
    start = time.time()
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
    rfecv.fit(x_train, y_train)
    n = rfecv.n_features_
    print("AdaBoost Classifier - RFECV")
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    feat_importances = pd.Series(rfecv.ranking_, index=x_train.columns)
    x = feat_importances.nsmallest(n).index
    x_train_new = x_train.loc[:,x]
    x_test_new = x_test.loc[:,x] 
    
    params={"n_estimators":[10, 50, 100, 500],
            "learning_rate" : [0.0001, 0.001, 0.01, 0.1, 1.0]}
    gs_adab = GridSearchCV(estimator = GradientBoostingClassifier(),
                          param_grid=params,
                          scoring='accuracy',
                          cv=5)
    gs_adab.fit(x_train_new, y_train)
    train_predictions = gs_adab.predict(x_test_new)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[7, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[7, 'MLA Time'] = end - start

def svc_1(x_train, y_train, x_test, y_test):
    MLA_compare.loc[8, 'MLA Name'] = "Support Vector Classifier, running on all features"
    model = SVC()
    start = time.time()
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_test)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[8, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[8, 'MLA Time'] = end - start

def svc_2(x_train, y_train, x_test, y_test):
    MLA_compare.loc[9, 'MLA Name'] = "Support Vector Classifier Classifier, Tuned and running on selected features"
    model = SVC()
    start = time.time()
    x_train_new, x_test_new = select_best_features(x_train, x_test, 30) 
    
    params = {'C': [0.1,1, 10, 100],
              'gamma': [1,0.1,0.01,0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']
             }
    gs_svc = GridSearchCV(estimator = SVC(),
                          param_grid=params,
                          scoring='accuracy',
                          cv=5)
    gs_svc.fit(x_train_new, y_train)
    train_predictions = gs_svc.predict(x_test_new)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[9, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[9, 'MLA Time'] = end - start

def xgb_1(x_train, y_train, x_test, y_test):
    MLA_compare.loc[10, 'MLA Name'] = "XGB Classifier, running on all features"
    model = XGBClassifier(silent=True)
    start = time.time()
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_test)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[10, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[10, 'MLA Time'] = end - start

def xgb_2(x_train, y_train, x_test, y_test):
    MLA_compare.loc[11, 'MLA Name'] = "XGB Classifier, Tuned and running on selected features"
    model = XGBClassifier(silent=True)
    start = time.time()
    
    #feature selection through RFECV
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
    rfecv.fit(x_train, y_train)
    n = rfecv.n_features_
    print("XGB Classifier - RFECV")
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    feat_importances = pd.Series(rfecv.ranking_, index=x_train.columns)
    x = feat_importances.nsmallest(n).index
    
    #new features are saved in x_train_new and x_test_new
    x_train_new = x_train.loc[:,x]
    x_test_new = x_test.loc[:,x] 
    
    params = {'learning_rate': [.01, .03, .05, .1, .25],
              'max_depth': [1,2,4,6,8,10],
              'n_estimators': [10, 50, 100, 300, 500], 
             }
    gs_svc = GridSearchCV(estimator = XGBClassifier(silent=True),
                          param_grid=params,
                          scoring='accuracy',
                          cv=5)
    gs_svc.fit(x_train_new, y_train)
    train_predictions = gs_svc.predict(x_test_new)
    acc = accuracy_score(y_test, train_predictions)
    end = time.time()

    MLA_compare.loc[11, 'MLA Accuracy'] = acc*100
    MLA_compare.loc[11, 'MLA Time'] = end - start

x_train_knn, x_test_knn = fnorm(x_train, x_test)
KNN_1(x_train_knn, y_train, x_test_knn, y_test)
KNN_2(x_train_knn, y_train, x_test_knn, y_test)
rf = rf_1(x_train, y_train, x_test, y_test)
gbc_1(x_train, y_train, x_test, y_test)
adab_1(x_train, y_train, x_test, y_test)
svc_1(x_train_knn, y_train, x_test_knn, y_test)
xgb_1(x_train, y_train, x_test, y_test)

rf_2(x_train, y_train, x_test, y_test)
gbc_2(x_train, y_train, x_test, y_test)
adab_2(x_train, y_train, x_test, y_test)
svc_2(x_train_knn, y_train, x_test_knn, y_test)
xgb_2(x_train, y_train, x_test, y_test)

print(MLA_compare)

confmat = confusion_matrix(y_true=y_test, y_pred=rf.predict(x_test))
print(confmat)

df_cm = pd.DataFrame(confmat, index = [i for i in ['Label 0','Label 1','Label 2']],
                    columns = [i for i in ["Predicted 0", "Predicted 1", "Predicted 2"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot = True)