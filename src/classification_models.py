from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "KNN Classifier": KNeighborsClassifier(),
        "SVC": SVC(),
        "XGBoost Classifier": XGBClassifier()
    }