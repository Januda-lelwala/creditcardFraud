import numpy as np
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


class XGBoostModel:
    def __init__(self, n_estimators=300, max_depth=17, learning_rate=0.06,
                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                 smote_random_state=28):
        self._smote_rs = smote_random_state
        self._n_estimators = n_estimators
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric='aucpr',
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X_train, y_train):
        print("  Applying SMOTE...", flush=True)
        X_sm, y_sm = SMOTE(random_state=self._smote_rs).fit_resample(X_train, y_train)
        print(f"  After SMOTE — fraud: {y_sm.sum():,}, normal: {(y_sm == 0).sum():,}", flush=True)
        print(f"  Training XGBoost ({self._n_estimators} rounds)...", flush=True)
        self._model.fit(X_sm, y_sm)
        print("  Done.", flush=True)
        return self

    def scores(self, X):
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self._model.predict(X)
