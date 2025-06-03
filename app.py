# app.py
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import soundfile as sf
from datetime import datetime
from preprocessing_model import SingleAudioProcessor
from pazhvak_hu.models.pazhvak_cnn import CNNLight, CNNHeavy
from pazhvak_hu.models.pazhvak_lstm import LSTMLight, LSTMHeavy
from pazhvak_hu.models.pazhvak_crnn import CRNN

MODEL_PATH = "pazhvak_hu\models\Best_Models"

# Load model and labels (cache these for performance)
@st.cache_resource
def load_model(model_name: str):
    model_info = torch.load(os.path.join(MODEL_PATH, model_name), map_location='cpu')
    if isinstance(model_info, dict) and 'model_state_dict' in model_info:
        # Determine input_channels from model type
        if "cnn_light" in model_name or "cnn_heavy" in model_name or "crnn" in model_name:
            input_channels = 1  # spectrograms usually have 1 channel
        elif "lstm" in model_name:
            input_channels = None  # LSTMs typically don't need input_channels like CNNs
        else:
            raise ValueError(f"Unknown model type in file: {model_name}")
        
        if "cnn_light" in model_name:
            model = CNNLight(input_channels)
        
        elif "cnn_heavy" in model_name:
            model = CNNHeavy(input_channels)
        
        elif "lstm_light" in model_name:
            model = LSTMLight()
        
        elif "lstm_heavy" in model_name:
            model = LSTMHeavy()
        
        elif "crnn" in model_name:
            model = CRNN(input_channels)
        
        else:
            raise ValueError(f"Unknown model type in file: {model_name}")
        
        model.load_state_dict(model_info['model_state_dict'])
    else:
        # Direct model (not a dict), likely saved via `torch.save(model, path)`
        model = model_info
        
    model.eval()  # Set to evaluation mode
    return model

@st.cache_resource
def load_labels():
    df = pd.read_excel('assets/labels.xlsx')
    return df

# Initialize audio processor
@st.cache_resource
def get_audio_processor():
    return SingleAudioProcessor()

# App title and description
st.title("Pazhvak Voice Classification")
st.write("""
Developing a Persian ASR model for 3602-word classification using the PAZHVAK dataset 
at University of Hormozgan. This app classifies Persian voice samples into one of 3602 categories.
""")

# Sidebar with info
st.sidebar.title("About")
st.sidebar.write("""
**Pazhvak Voice Classification**  
GitHub: [Github Repository](https://github.com/Sepi1010011/PAZHVAK_Speech_Recognition_HU/)  
Developed by University of Hormozgan Student  
""")

# Model selection dropdown
MODEL_OPTIONS = {
    "CNN Light": {"feature": "mel", "model_name": "cnn_light_best.pth"},
    "CNN Heavy": {"feature": "mel", "model_name": "cnn_heavy_best.pth"},
    "LSTM Light": {"feature": "mfcc", "model_name": "lstm_light_best.pth"},
    "LSTM Heavy": {"feature": "mfcc", "model_name": "lstm_heavy_best.pth"}, 
    "CRNN": {"feature": "mel", "model_name": "crnn_best.pth"}
}

selected_model = st.selectbox(
    "Select Model Architecture",
    list(MODEL_OPTIONS.keys()),
    index=0,
    help="CNN/CRNN models use Mel spectrograms, LSTM use MFCC features"
)

# Create two columns
col1, col2 = st.columns([0.6, 0.4], gap="small")

# Left column for file upload
with col1:
    st.subheader("Upload Voice Sample")
    uploaded_file = st.file_uploader("Choose a file", type=['wav', 'mp3', 'ogg'], label_visibility="collapsed")

# Right column for results
with col2:
    st.subheader("Classification Results")
    result_box = st.empty()
    result_box.info("Results will appear here after processing")
    
    st.caption(f"Selected Model: {selected_model}")
    st.caption(f"Feature Extraction: {MODEL_OPTIONS[selected_model]['feature'].upper()}")
    
    flag_button = st.empty()
    feedback_area = st.empty()

# Process file if uploaded
if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = f"temp_audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Display audio player
    with col1:
        st.audio(uploaded_file, format='audio/wav')
    
    # Get selected model configuration
    model_config = MODEL_OPTIONS[selected_model]
    
    with st.spinner(f'Processing audio with {selected_model} ({model_config["feature"].upper()} features)...'):
        try:
            # Initialize processors
            audio_processor = get_audio_processor()
            labels_df = load_labels()
            model = load_model(model_config["model_name"])
            
            # Preprocess and extract features
            features, audio = audio_processor.process_audio(
                file_path, 
                feature_type=model_config["feature"]
            )
            
            # Convert to tensor
            features_tensor = torch.from_numpy(features).float()
            
            # Predict
            with torch.no_grad():
                outputs = model(features_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()
            
                        
            print(predicted_class)
            # Get label
            label = labels_df[labels_df['Folder Number'] == predicted_class]['Persian Word'].values[0]
            
            # Display results
            with col2:
                result_box.success(f"""
                **Prediction Results:**
                - Model Used: {selected_model}
                - Features: {model_config["feature"].upper()}
                - Class ID: {predicted_class}
                - Label: {label}
                """)
                
                # Flagging system
                if flag_button.button("🚩 Flag Incorrect Classification", 
                                    type="primary"):
                    with st.form("feedback_form"):
                        user_feedback = st.text_area(
                            "Please describe the issue:",
                            placeholder="What was wrong with this classification?"
                        )
                        submitted = st.form_submit_button("Submit Feedback")
                        if submitted:
                            flag_data = {
                                'timestamp': datetime.now(),
                                'model_used': selected_model,
                                'features_used': model_config["feature"],
                                'file_name': uploaded_file.name,
                                'predicted_class': predicted_class,
                                'predicted_label': label,
                                'user_feedback': user_feedback
                            }
                            
                            flag_df = pd.DataFrame([flag_data])
                            header = not os.path.exists('flagged_data.csv')
                            flag_df.to_csv('flagged_data.csv', mode='a', header=header, index=False)
                            st.toast("Thank you for your feedback!", icon="👍")
                            flag_button.empty()
                            feedback_area.empty()
            
        except Exception as e:
            with col2:
                result_box.error(f"Error processing file: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)