# Hugging Face Deployment

## 1. Target Setup

Use a Hugging Face Space configured for Docker. The included [Dockerfile](../Dockerfile) starts the FastAPI app on port `7860`, which matches the Spaces runtime.

## 2. What Ships With the App

- FastAPI backend and static browser UI.
- Trained digit classifier and rejector artifacts under `artifacts/`.
- Runtime metrics files used by the local loader.

The live Space does not need to retrain on boot. The local `mnist.npz` dataset remains available for contributors who want to retrain the model outside the Space.

## 3. Deployment Steps

1. Create a new Hugging Face Space and choose Docker as the SDK.
2. Push this repository to the Space.
3. Wait for the build to finish and open the Space URL.
4. Validate the canvas, prediction endpoint, rejector status, and history panel in the browser.

## 4. Operational Notes

- The app serves the frontend and API from the same process.
- The `PORT` environment variable is respected, so the app can also run behind a different container port if needed.
- Browser/profile caches and temporary files stay out of the image through [`.dockerignore`](../.dockerignore).