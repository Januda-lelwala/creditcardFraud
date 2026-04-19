from src.data import load_and_preprocess
from src.evaluate import evaluate, save_results
from src.models.autoencoder import AutoencoderModel

X_train, X_test, y_train, y_test = load_and_preprocess()

print("\n--- Training Autoencoder ---")
model = AutoencoderModel(epochs=20)
model.fit(X_train, y_train)
model.save('models/autoencoder.pth')

scores = model.scores(X_test)
preds  = model.predict(X_test)
metrics = evaluate('Autoencoder', scores, preds, y_test)
save_results('Autoencoder', metrics, scores, y_test)
