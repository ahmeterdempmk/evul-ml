import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from metrics import get_metrics

class ModelSelector:
    def __init__(self, models):
        self.models = models

    def find_best_regressor(self, X_train, y_train, X_test, y_test):
        best_model = None
        best_score = -np.inf
        results = []

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_r2 = get_metrics(y_test, y_pred)['r2']
            results.append((name, test_r2))

            print(f"Model: {name.upper()}")
            print(f"Test R-Squared Score: {test_r2:.5f}\n")

            # 10-fold cross-validation
            scores = cross_validate(model, X_train, y_train,
                                    scoring=['r2', 'neg_mean_absolute_error', 
                                             'neg_mean_squared_error', 'neg_mean_absolute_percentage_error'],
                                    cv=10, return_train_score=False)

            scores_df = pd.DataFrame(scores, index=range(1, 11))
            avg_scores = scores_df.mean().abs().apply("{:.5f}".format)
            print(avg_scores)
            print("\n############################################################################\n")

            # Save best model based on R-Squared
            if test_r2 > best_score:
                best_score = test_r2
                best_model = (name, model)

        print(f"Best Regressor: {best_model[0]} with R-Squared: {best_score:.5f}")
        return best_model[0], best_model[1]

    def find_best_classifier(self, X_train, y_train, X_test, y_test):
        pass