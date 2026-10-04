
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from utils.utils import drop_outlier_rows


class TrainModel():
    # Note: this class does no Streamlit output so that app.py can cache it.
    def __init__(self, data, target, features):

        # Categorical features are used as given (integer codes), not one-hot encoded,
        # because the descriptions and categorical interpretations are keyed by feature name.
        self.features = list(features)
        # Remove rows with statistical outliers (IQR method) before training
        n_before = len(data)
        data, self.dropped_counts = drop_outlier_rows(data, columns=self.features)
        self.rows_removed = n_before - len(data)
        X = data[self.features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = sm.add_constant(X_train)
        self.X_test = sm.add_constant(X_test)
        self.y_train = y_train
        self.y_test = y_test
        self.elimination_log = []

        self._fit(self.features)

    def _fit(self, features):
        result = sm.Logit(self.y_train, self.X_train[['const'] + features]).fit(disp=False)

        # Extract coefficients and put them in a DataFrame
        coef_df = result.params.reset_index()
        coef_df.columns = ['Parameter', 'Value']
        # rename feature const to intercept
        coef_df['Parameter'] = coef_df['Parameter'].replace('const', 'Intercept')
        self.coef_df = coef_df
        self.p_values = result.pvalues.rename({'const': 'Intercept'})
        self.summary = result.summary()
        # Out-of-sample discrimination on the held-out 20%
        self.test_auc = roc_auc_score(self.y_test, result.predict(self.X_test[['const'] + features]))
        return result

    def selectFeatures(self, significance_level=0.05):
        # Backward elimination
        current_features = list(self.features)
        while len(current_features) > 1:
            model = sm.Logit(self.y_train, self.X_train[['const'] + current_features]).fit(disp=False)

            # Get the predictor with the highest p-value (exclude intercept)
            p_values = model.pvalues.drop('const')
            highest_p_value = p_values.max()
            feature_to_remove = p_values.idxmax()

            if highest_p_value > significance_level:
                self.elimination_log.append((feature_to_remove, highest_p_value))
                current_features.remove(feature_to_remove)  # Remove the predictor
            else:
                break  # Stop if all p-values are below the significance level

        # Final model with selected features
        self.features = current_features
        self._fit(current_features)
