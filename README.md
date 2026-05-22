---
title: Handwritten Digit Recognition
emoji: ✍️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: 'A handwritten digit recognizer'
---

# Handwritten Digit Recognition

A single FastAPI app with a custom HTML/CSS/JS frontend for recognizing handwritten digits from a browser canvas. The same codebase serves the UI, the prediction API, shared preprocessing, and model training.

## Project Status

This project is in a usable local-demo state and is organized as one deployable service rather than separate frontend and backend apps.

- Browser-based drawing canvas with pen, eraser, undo, redo, and brush size controls
- Multi-digit segmentation with clickable region switching
- `/api/predict` endpoint for live inference from the canvas image
- Shared preprocessing pipeline for both training and inference
- Plain MNIST-trained `MLPClassifier(hidden_layer_sizes=(256, 128))` classifier plus a top-level binary rejector
- Probability breakdown, recent prediction history, and preview rendering in the UI
- Pytest coverage for dataset loading, preprocessing, inference, and web routes

## How It Works

1. The frontend sends a base64-encoded canvas image to the FastAPI backend.
2. `digit_app.preprocess` normalizes the image, detects foreground regions, and prepares MNIST-like input.
3. `digit_app.inference` loads the saved digit classifier, predicts each detected region, and returns structured results.
4. `digit_app.service` applies the rejector model on top of the classifier output and labels uncertain or non-digit drawings.
5. `digit_app.web` serves the dashboard and returns prediction payloads that the browser renders immediately.

## Repository Layout

```text
app.py          Local entrypoint for the FastAPI app
digit_app/      Training, preprocessing, inference, configuration, and API code
web/            Static frontend assets
tests/          Automated tests
docs/           Design and reference material
requirements.txt Runtime and test dependencies
mnist.npz       Local MNIST dataset used for training
```

## Local Setup

The project expects a local `mnist.npz` file at the repository root. Keep it local if you do not want to commit the dataset.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the Model

Train the MNIST classifier and generate `artifacts/model.joblib` plus the metrics files:

```powershell
python -m digit_app.train
```

Train the rejector model after the classifier exists:

```powershell
python -m digit_app.rejector
```

## Run the App

Start the FastAPI server locally:

```powershell
python app.py
```

Open `http://127.0.0.1:8000` in your browser.

If port `8000` is already in use, set a different port before launching:

```powershell
$env:PORT = "8010"
python app.py
```

If `artifacts/model.joblib` is missing, the backend will train a model automatically on first launch.

## Deploy to Hugging Face Spaces

This repository is ready for a Docker Space.

- The container serves the FastAPI app on port `7860`.
- The trained classifier, rejector, and metrics files are versioned so the app starts with working inference immediately.
- The local `mnist.npz` dataset is still useful if you want to retrain outside the Space, but the live Space does not depend on it.
- Use the included [Dockerfile](Dockerfile) and push the repository to a new Hugging Face Space configured for Docker.

## Model

The digit classifier is defined in [digit_app/train.py](digit_app/train.py#L39). The rejector model is defined in [digit_app/rejector.py](digit_app/rejector.py#L1).

- `scikit-learn` `MLPClassifier`
- Hidden layers: `(256, 128)`
- `ReLU` activation
- `Adam` optimizer
- Early stopping for faster and more stable local training
- `LogisticRegression` rejector on top of digit probabilities and crop statistics

## Testing

Run the full test suite with:

```powershell
python -m pytest
```

## Deployment Notes

The app is designed to run as one web service, so it is a good fit for a single-container deployment.

- Set `HOST=0.0.0.0` and use the platform-provided `PORT`.
- Include the trained model artifact if you want faster startup.
- Keep `mnist.npz` out of the deployment image unless you want to retrain in the host environment.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.

## Notes

- Generated files such as browser/profile caches, `.tmp/`, `.pytest_cache/`, and virtual environments are excluded through `.gitignore`.
- The deployable model artifacts live in `artifacts/` so the Space can boot without retraining.
- Design notes and deployment guidance live in `docs/` so the repository root stays focused on runnable source code.



