import gradio as gr
import numpy as np
import joblib
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("App ready — model and scaler will be loaded based on user selection.")

# === Load scaler for age normalization (must match training!)
try:
    age_scaler = joblib.load("age_scaler.pkl")
    print("✅ Age scaler loaded.")
except Exception as e:
    age_scaler = None
    print("⚠️ Warning: age_scaler.pkl not found. Raw age will be used!")

def process_image(image, size=(64, 64)):
    """Convert image to 64x64 grayscale and flatten to 4096-length vector."""
    if image is None:
        raise ValueError("No image provided")

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    img = image.convert("L").resize(size)
    return np.array(img).flatten()

def predict_mortality(image, age, model_choice="best_xgb.pkl"):
    """Predict time to death using image and age input."""
    try:
        model = joblib.load(model_choice)

        if image is None:
            return {
                "Error": "No image uploaded",
                "Note": "Please upload a face image"
            }

        if age is None or age < 0 or age > 120:
            return {
                "Error": "Invalid age",
                "Note": "Please enter a valid age between 0 and 120"
            }

        # Image to 4096 features
        img_features = process_image(image)

        # Age scaling (required to match training input distribution)
        if age_scaler is not None:
            scaled_age = age_scaler.transform(np.array(age).reshape(1, -1))[0, 0]
        else:
            scaled_age = age  # fallback if scaler is missing

        gender_encoded = 0  # gender not used, default to 0

        # Combine image + scaled age + gender = 4098 features
        base_features = np.concatenate([img_features, [scaled_age, gender_encoded]])

        # Pad if needed (for models expecting COD features too)
        model_input_dim = getattr(model, "n_features_in_", len(base_features))
        if len(base_features) > model_input_dim:
            raise ValueError(f"Too many features: got {len(base_features)}, model expects {model_input_dim}")
        elif len(base_features) < model_input_dim:
            padding = np.zeros(model_input_dim - len(base_features))
            features = np.concatenate([base_features, padding])
        else:
            features = base_features

        prediction = model.predict(features.reshape(1, -1))[0]
        life_expectancy = age + prediction

        return {
            "Model file": model_choice,
            "Predicted time to death": f"{prediction:.1f} years",
            "Estimated life expectancy": f"{life_expectancy:.1f} years",
            "Current age": f"{age} years"
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error occurred: {error_details}")
        return {
            "Error": str(e),
            "Details": error_details.split('\n')[-3:-1],
            "Note": "Please ensure you've uploaded a clear face image"
        }

# === Gradio Interface ===
iface = gr.Interface(
    fn=predict_mortality,
    inputs=[
        gr.Image(type="pil", label="Face Image"),
        gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Age"),
        gr.Radio(
            choices=["best_xgb.pkl", "reg_xgb_ttd.pkl", "reg_rf_ttd.pkl"],
            value="best_xgb.pkl",
            label="Select Model File"
        )
    ],
    outputs=gr.JSON(label="Prediction Results"),
    title="Face Mortality Prediction",
    description="""
    This model predicts life expectancy based on facial features and age.
    
    **Important Notes:**
    - This version correctly scales age input to match training
    - Upload a clear, front-facing photo for best results
    - Models were trained with standardized age inputs
    """,
    article="""
    ## Model Options
    
    - **best_xgb.pkl**: Trained with image + metadata + cause-of-death (4436 features)
    - **reg_xgb_ttd.pkl**: Regression-only model (image + age, 4098 features)
    - **reg_rf_ttd.pkl**: Random Forest regression (image + age, 4098 features)

    Age input is automatically standardized using the training `StandardScaler`.

    ### Disclaimer
    This tool is for research purposes only.
    """
)

if __name__ == "__main__":
    iface.launch()