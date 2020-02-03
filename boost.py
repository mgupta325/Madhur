import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, datasets,svm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.2,1.0,10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print('cross-validation scores', test_scores_mean)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt


def plot_validation_curve(X,y,estimator,name):

    if name == 'k1':
        param_range = np.arange(20,50,5)
        param_name = 'n_estimators'
    if name == 'k2':
          param_range = np.arange(2,14,2)
          param_name = 'max_depth'


    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=5,
                                                 scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve for " + name)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.grid()
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


df = pd.read_csv('digits_training.tra', sep=",", skiprows=0)

df=np.array(df)
print(df.shape,df.dtype)
dat=df[:,0:64]
tar1=df[:,64]
X=dat
y=tar1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=20)

boostc = GradientBoostingClassifier(n_estimators=30,max_depth=6)
s1=time.time()
boostc.fit(X_train,y_train)
e1=time.time()
# s2=time.time()
# y_prd=boostc.predict(X_test)
# e2=time.time()
t1=e1-s1
#t2=e2-s2
y_test = np.array(y_test)
#y_prd = np.array(y_prd)
print(boostc.get_params())
#print('Test Accuracy: %.8f' % accuracy_score(y_test,y_prd))
print("Training time: ",t1)
#print("Testing time: ",t2)
plot_learning_curve(boostc,'Learning Curve for boosting', X_train, y_train, (0,1.01),cv=5)

boostc1 = GradientBoostingClassifier()
plot_validation_curve(X_train,y_train,boostc1,'k1')
plot_validation_curve(X_train,y_train,boostc1,'k2')
#plot_validation_curve(X_train,y_train,boostc1,'k3')
# plot_validation_curve(X_train,y_train,clf1,'dtree3')


boostc3 = GradientBoostingClassifier()
boostc3=GridSearchCV(estimator=boostc3,param_grid={'n_estimators':np.arange(20,40,10),'learning_rate':[0.5],'max_depth':np.arange(4,8,2),'max_leaf_nodes':[180,250]},cv=5)
x1=time.time()
boostc3.fit(X_train,y_train)
x2=time.time()
print(x2-x1,"training time")
x11=time.time()
y_prd1=boostc3.predict(X_test)
x22=time.time()
print(x22-x11,"testing time")
y_prd1 = np.array(y_prd1)
print('Test Accuracy after grid search fit: %.8f' % accuracy_score(y_test,y_prd1))
print(boostc3.best_params_,boostc3.best_score_)
boostc4 = GradientBoostingClassifier(max_depth=4,max_leaf_nodes=200,n_estimators=20,ccp_alpha=0.0001)
boostc4.fit(X_train,y_train)
plot_learning_curve(boostc4,'Learning Curve for boosting', X_train, y_train, (0,1.01),cv=5)

#if learning curve take too much time then create clf4 with best parameters as arguments



