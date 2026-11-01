import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def create_new_features(X):
    x = X.copy()
    x['sqrt_app_crashes_last_month'] = np.sqrt(x['app_crashes_last_month'].clip(lower=0))
    x['square_days_since_last_promo'] = x['days_since_last_promo'] ** 2
    x['sqrt_order_frequency_month'] = np.sqrt(x['order_frequency_month'].clip(lower=0))
    x['crashes_per_open'] = x['app_crashes_last_month'] / (4*x['app_opens_per_week'].clip(lower=0) + 1)
    x['app_engagement'] = x['app_opens_per_week'] / (x['order_frequency_week'].clip(lower=0) + 1)
    x['total_spent_last_month'] = np.sqrt(x['total_spent_last_month'].clip(lower=0))
    
    return x


class ChurnPredictorStrategy(BaseEstimator, ClassifierMixin):
    
    def __init__(self, pipeline, strategy='max_f1_score'):
        self.pipeline = pipeline
        self.strategy = strategy
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        y_pred_proba = self.pipeline.predict_proba(X)[:, 1]
        
        if self.strategy == 'max_recall':
            predictions = (y_pred_proba > 0.07).astype(int)
        elif self.strategy == 'max_precision':
            predictions = (y_pred_proba > 0.81).astype(int)
        else:
            predictions = (y_pred_proba > 0.43).astype(int)
        
        return predictions
