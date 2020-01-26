import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, datasets,svm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
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
        param_range = np.arange(10,100,10)
        param_name = 'n_estimators'
    # if name == 'k2':
    #      param_range = np.arange(2,35,2)
    #      param_name = 'min_samples_split'


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


df = pd.read_csv('tic-tac-toe.data', sep=",", skiprows=0)
print(df.head(10))
df=np.array(df)
print(df.shape,df.dtype)
dat=df[:,0:9]
tar1=df[:,9]
X=dat
y=tar1
print(y.dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20)
clf=DecisionTreeClassifier()
boostc = AdaBoostClassifier(base_estimator=clf)
s1=time.time()
boostc.fit(X_train,y_train)
e1=time.time()
s2=time.time()
y_prd=boostc.predict(X_test)
e2=time.time()
t1=e1-s1
t2=e2-s2
y_test = np.array(y_test)
y_prd = np.array(y_prd)
print(boostc.get_params())
print('Test Accuracy: %.8f' % accuracy_score(y_test,y_prd))
print("Training time: ",t1)
print("Testing time: ",t2)
plot_learning_curve(boostc,'Learning Curve for boosting', X_train, y_train, (0,1.01),cv=5)
# plot_tree(boostc.fit(X_train, y_train),filled=True)
# plt.show()
clf1=DecisionTreeClassifier(max_depth=10,max_leaf_nodes=100)
boostc1 = AdaBoostClassifier(base_estimator=clf1)
plot_validation_curve(X_train,y_train,boostc1,'k1')
# plot_validation_curve(X_train,y_train,clf1,'dtree1')
# #plot_validation_curve(X_train,y_train,clf1,'dtree2')
# plot_validation_curve(X_train,y_train,clf1,'dtree3')

x1=time.time()
clf3=DecisionTreeClassifier(max_depth=8,max_leaf_nodes=80,ccp_alpha=0.01)
boostc3 = AdaBoostClassifier(base_estimator=clf3)
boostc3=GridSearchCV(estimator=boostc3,param_grid={'n_estimators':np.arange(30,60,10),'learning_rate':[1,1.2]},cv=5)
x2=time.time()
print(x2-x1)
boostc3.fit(X_train,y_train)
# #  plot_tree(boostc1.fit(X_train, y_train),filled=True)
# #  plt.show()
y_prd1=boostc3.predict(X_test)
y_prd1 = np.array(y_prd1)
print('Test Accuracy after grid search fit: %.8f' % accuracy_score(y_test,y_prd1))
print(boostc3.best_params_,boostc3.best_score_)
plot_learning_curve(boostc3,'Learning Curve for boosting', X_train, y_train, (0,1.01),cv=5)

#if learning curve take too much time then create clf4 with best parameters as arguments




