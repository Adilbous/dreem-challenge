# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier


class CompareClassifiers:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def run(self):
        # prepare configuration for cross validation test harness
        seed = 7
        # prepare models
        models = []
        models.append(('LR', LogisticRegression(random_state=0, solver='lbfgs')))
        models.append(('BAG', BaggingClassifier(n_estimators=20, n_jobs=-1)))
        models.append(('GB', GradientBoostingClassifier(n_estimators=20,)))
        models.append(('SVC', SVC(gamma='auto')))
        models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=0, n_jobs = -1, bootstrap=True, max_features='sqrt')))
        
        # evaluate each model in turn
        results = []
        names = []
        scoring = 'f1_macro'
        for name, model in models:
            print("running for %s" % (name))
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, self.X, self.y, cv=kfold, scoring=scoring, verbose=2)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        ax.set(ylabel='F1 score')
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()
        fig.savefig('compare_clf.png')