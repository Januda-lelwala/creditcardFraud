from src.data import load_and_preprocess
from src.evaluate import evaluate, save_results
from src.models.xgboost_model import XGBoostModel

X_train, X_test, y_train, y_test = load_and_preprocess()

print("\n--- Training XGBoost + SMOTE ---")
model = XGBoostModel()
model.fit(X_train, y_train)

scores = model.scores(X_test)
preds  = model.predict(X_test)
metrics = evaluate('XGBoost + SMOTE', scores, preds, y_test)
save_results('XGBoost + SMOTE', metrics, scores, y_test)
