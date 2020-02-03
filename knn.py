import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, datasets,svm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
import itertools
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
        param_range = np.arange(1, 12,1)
        param_name = 'n_neighbors'
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


df = pd.read_csv('digits_training.tra', sep=",", skiprows=0)

df=np.array(df)
print(df.shape,df.dtype)
dat=df[:,0:64]
tar1=df[:,64]
X=dat
y=tar1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20)
clf=KNeighborsClassifier()
s1=time.time()
clf.fit(X_train,y_train)
e1=time.time()
t1=e1-s1
y_test = np.array(y_test)
print(clf.get_params())
print("Training time: ",t1)

plot_learning_curve(clf,'Learning Curve for K-nn', X_train, y_train, (0,1.01),cv=5)
# plot_tree(clf.fit(X_train, y_train),filled=True)
# plt.show()
clf1=KNeighborsClassifier()
plot_validation_curve(X_train,y_train,clf1,'k1')


clf3=KNeighborsClassifier()
clf3=GridSearchCV(estimator=clf3,param_grid={'n_neighbors':np.arange(2,8,1),'weights':['uniform','distance']},cv=5)
x1=time.time()
clf3.fit(X_train,y_train)
x2=time.time()
print(x2-x1,"train time")
x11=time.time()
y_prd1=clf3.predict(X_test)
x21=time.time()
print(x21-x11,"test time")
y_prd1 = np.array(y_prd1)
print('Test Accuracy after grid search fit: %.8f' % accuracy_score(y_test,y_prd1))
print(clf3.best_params_,clf3.best_score_)
clf4=KNeighborsClassifier(n_neighbors=6,weights='distance')
clf4.fit(X_train,y_train)
plot_learning_curve(clf4,'Learning Curve for K-nn', X_train, y_train, (0,1.01),cv=5)





