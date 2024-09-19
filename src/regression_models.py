from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def get_regressors():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Elastic Net Regression": ElasticNet(),
        "Random Forest Regression": RandomForestRegressor(),
        "Gradient Boosting Regression": GradientBoostingRegressor(),
        "Bayesian Regression": BayesianRidge(),
        "SVR": SVR(),
        "Decision Tree Regression": DecisionTreeRegressor(),
        "KNN Regression": KNeighborsRegressor(),
        "XGBRegressor": XGBRegressor()
    }