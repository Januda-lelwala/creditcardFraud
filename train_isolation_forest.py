from src.data import load_and_preprocess
from src.evaluate import evaluate, save_results
from src.models.isolation_forest import IsolationForestModel

X_train, X_test, y_train, y_test = load_and_preprocess()

print("\n--- Training Isolation Forest ---")
model = IsolationForestModel()
model.fit(X_train, y_train)

scores = model.scores(X_test)
preds  = model.predict(X_test)
metrics = evaluate('Isolation Forest', scores, preds, y_test)
save_results('Isolation Forest', metrics, scores, y_test)
