import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)


# ==================================================
# Custom CSS
# ==================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    background-color: #f0f2f6;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# Load Model
# ==================================================

@st.cache_resource
def load_emotion_model():

    model = tf.keras.models.load_model(
        "emotion_detection_final.keras"
    )

    return model


model = load_emotion_model()


# ==================================================
# Emotion Labels
# ==================================================

EMOTION_LABELS = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise"
]


# ==================================================
# Title
# ==================================================

st.markdown(
    '<div class="main-title">😊 Emotion Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning based Facial Emotion Recognition'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# Upload Image
# ==================================================

st.subheader("Upload a Facial Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# Prediction Function
# ==================================================

def predict_emotion(image):

    # Convert to grayscale
    image = image.convert("L")

    # Resize to model input size
    image = image.resize((48, 48))

    # Convert to NumPy
    image_array = np.array(
        image
    ).astype(np.float32)

    # Normalize exactly like training
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Add channel dimension
    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Get probabilities
    probabilities = predictions[0]

    # Get predicted class
    predicted_index = np.argmax(
        probabilities
    )

    predicted_emotion = EMOTION_LABELS[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ]

    return (
        predicted_emotion,
        confidence,
        probabilities
    )


# ==================================================
# Display Image + Prediction
# ==================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    # Predict button
    if st.button(
        "🔍 Detect Emotion",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing the image..."
        ):

            emotion, confidence, probabilities = \
                predict_emotion(image)


        # ==========================================
        # Result
        # ==========================================

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            f"Detected Emotion: {emotion}"
        )

        st.write(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ==========================================
        # Probability Chart
        # ==========================================

        st.subheader(
            "Emotion Probabilities"
        )

        probability_data = {
            EMOTION_LABELS[i]:
            float(probabilities[i])
            for i in range(
                len(EMOTION_LABELS)
            )
        }

        st.bar_chart(
            probability_data
        )


# ==================================================
# Model Information
# ==================================================

st.divider()

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Input Size",
        "48 × 48"
    )

with col2:

    st.metric(
        "Test Accuracy",
        "62.47%"
    )

st.caption(
    "Model trained using CNN with data augmentation, "
    "hyperparameter tuning, Early Stopping and "
    "ReduceLROnPlateau."
)