---
title: Face Mortality Prediction
emoji: 🔮
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---

# Face Mortality Prediction

This space hosts an XGBoost model that predicts life expectancy based on facial features and age.

## Model Information

- **Model Type**: XGBoost
- **Feature Extraction**: FaceNet (InceptionResnetV1)
- **Input Features**: Face image (converted to 4096-dim embedding) + age
- **Output**: Predicted years until death / life expectancy

## Important Disclaimer

⚠️ **This is a research model and should NOT be used for medical decisions or life planning.**

The model was trained on historical data from the IMDB-WIKI dataset and is intended for research purposes only.

## How it Works

1. Upload a clear, front-facing face photo
2. Enter the person's current age
3. The model extracts facial features using FaceNet
4. XGBoost predicts the remaining years of life
5. Life expectancy is calculated as current age + predicted years

## Technical Details

- Face images are resized to 160x160 pixels
- FaceNet extracts 4096-dimensional feature vectors
- The model was trained without gender information (due to missing data)
- Cause of death features are padded with zeros for inference

## Citation

If you use this model in research, please cite the original IMDB-WIKI dataset and the face mortality prediction work.