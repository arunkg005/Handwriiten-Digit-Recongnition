# Digit Recognition Architecture

## 1. Goal

Provide a single FastAPI service that lets a user draw handwritten digits in the browser, classifies each detected region, and flags uncertain or non-digit input with a rejector layer.

## 2. End-to-End Flow

1. The user opens the web app and draws on the canvas.
2. The frontend sends the canvas image to the API.
3. `digit_app.preprocess` normalizes the image, segments foreground regions, and prepares MNIST-shaped inputs.
4. `digit_app.inference` runs the trained `MLPClassifier` and produces per-region probabilities.
5. `digit_app.service` applies geometry checks and the binary rejector to gate uncertain results.
6. `digit_app.web` returns the final payload to the browser, which updates the prediction, probability breakdown, preview, and history.

## 3. Components

### Frontend

- Drawing canvas with pen, eraser, undo, redo, and brush size controls.
- Live prediction panel with status, confidence, probability breakdown, and recent history.
- Region switching for multi-digit drawings.

### Backend

- FastAPI application serving both the API and the static dashboard.
- Shared preprocessing used by training and inference.
- On-demand loading of the classifier and rejector artifacts.

### Model Stack

- `scikit-learn` `MLPClassifier` with hidden layers `(256, 128)`.
- `LogisticRegression` rejector trained on classifier probabilities and crop statistics.
- Geometry-based rejection checks for obviously non-digit marks.

## 4. Data Flow

- Train on the local MNIST dataset.
- Save the classifier, rejector, and evaluation metrics under `artifacts/`.
- Serve browser drawings through the same preprocessing path used during training.
- Return the active digit, confidence, uncertainty status, and region metadata to the UI.

## 5. Operational Notes

- The service can auto-train missing artifacts on first launch during local development.
- For Hugging Face Spaces, the trained artifacts are shipped with the repository so the app boots immediately.
- The browser UI is intended for repeated digit entry without reloading.
- The rejector exists to keep low-confidence marks from being presented as definitive digits.

## 6. Deliverables

- Training pipeline.
- Inference and service layer.
- Browser UI.
- Evaluation metrics and saved artifacts.
