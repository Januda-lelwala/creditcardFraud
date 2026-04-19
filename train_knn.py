from src.data import load_and_preprocess
from src.evaluate import evaluate, save_results
from src.models.knn import KNNModel

X_train, X_test, y_train, y_test = load_and_preprocess()

print("\n--- Training KNN + SMOTE ---")
model = KNNModel()
model.fit(X_train, y_train)

scores = model.scores(X_test)
preds  = model.predict(X_test)
metrics = evaluate('KNN + SMOTE', scores, preds, y_test)
save_results('KNN + SMOTE', metrics, scores, y_test)
