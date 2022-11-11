from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

def svmNuClassifier(features, labels):
    # clf = svm.NuSVC(gamma="auto")
    parameters = {'kernel': (
        "linear", "poly", "rbf", "sigmoid"), 'nu': [0.4, 0.5, 0.6]}
    nusvc = svm.NuSVC()
    clf = make_pipeline(StandardScaler(), GridSearchCV(nusvc, parameters))
    clf.fit(features, labels)
    return clf


def adaboostClassifier(features, labels):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(features, labels)
    return clf


def randomForestClassifier(features, labels):
    parameters = {'n_estimators': [10,100,1000], 'max_features': ["auto", "sqrt", "log2"]}
    mlp = RandomForestClassifier()
    clf = make_pipeline(StandardScaler(), GridSearchCV(mlp, parameters))
    clf.fit(features, labels)
    return clf
