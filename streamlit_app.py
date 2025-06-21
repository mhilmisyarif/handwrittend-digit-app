import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Function to load the trained Generator model ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_generator_model():
    try:
        # Ensure the path matches where you saved the model in Part 1
        model_path = './cgan_generator_mnist.h5'
        # Important: When loading a custom model, ensure it's built or compile if needed.
        # For pure inference, load_model is usually sufficient if saved correctly.
        model = tf.keras.models.load_model(model_path)
        # Optional: Print model summary to verify its structure after loading
        # model.summary()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the CGAN training script (Part 1) has been run successfully "
                "and 'cgan_generator_mnist.h5' exists in the same directory.")
        st.info("If the file exists, the error might be due to a TensorFlow/Keras version mismatch "
                "between where the model was saved and where it's being loaded.")
        return None

# Load the generator model
generator = load_generator_model()

if generator is None:
    st.stop() # Stop the app if model loading failed

# --- 2. Streamlit UI Setup ---
st.set_page_config(
    page_title="Handwritten Digit Image Generator",
    page_icon="✍️",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .stSelectbox {
        border-radius: 8px;
        box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 8px;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #333;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    p {
        color: #555;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("✍️ Handwritten Digit Image Generator")

st.markdown(
    "Generate synthetic MNIST-like images using your trained Conditional GAN model."
)

# --- 3. User Input for Digit Selection ---
col1, col2 = st.columns([1, 2])

with col1:
    selected_digit = st.selectbox(
        "Choose a digit to generate (0-9):",
        options=list(range(10)), # Dropdown from 0 to 9
        index=2 # Default to digit 2 as per example image
    )

with col2:
    st.write("") # Spacer for alignment
    st.write("") # Spacer for alignment
    num_images_to_generate = st.slider("Number of images to generate:", min_value=1, max_value=10, value=5)


if st.button("Generate Images"):
    if generator is not None:
        with st.spinner(f"Generating {num_images_to_generate} images of digit {selected_digit}..."):
            # Generate random noise for the number of images requested
            noise = tf.random.normal([num_images_to_generate, 100]) # 100 is NOISE_DIM

            # Create one-hot encoded labels for the selected digit
            # Repeat the one-hot vector for the number of images to generate
            labels = tf.keras.utils.to_categorical([selected_digit] * num_images_to_generate, num_classes=10)

            # --- FIX: Removed training=False from the generator call ---
            generated_images = generator([noise, labels])

            # Rescale images from [-1, 1] to [0, 1] for displaying
            generated_images = (generated_images * 0.5) + 0.5

            st.subheader(f"Generated images of digit {selected_digit}")

            # Display generated images
            cols = st.columns(num_images_to_generate)
            for i in range(num_images_to_generate):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(2, 2)) # Smaller figure size
                    ax.imshow(generated_images[i, :, :, 0], cmap='gray')
                    ax.axis('off')
                    ax.set_title(f"Sample {i+1}", fontsize=8) # Add sample label
                    st.pyplot(fig) # Display the plot in Streamlit
                    plt.close(fig) # Close the figure to prevent memory issues

    else:
        st.warning("Generator model not loaded. Please check the console for errors.")

st.markdown("---")
st.markdown("Model trained from scratch using TensorFlow/Keras and deployed with Streamlit.")
