

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np
import librosa
from PIL import Image
from moviepy.editor import VideoFileClip
import os
import tempfile
import gc
from transformers import ViTModel


# Define the combined ViT and StyleGAN model class
class ViT_StyleGAN_Model(nn.Module):
    def __init__(self, num_classes):
        super(ViT_StyleGAN_Model, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.vit(pixel_values=x).last_hidden_state[:, 0, :]  # Extract the [CLS] token
        features = features.view(batch_size, seq_len, -1)
        out = self.fc(features[:, -1, :])  # Use the output of the last sequence element
        return out

# Define the ViT model for audio
class ViTDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTDeepfakeDetector, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        num_features = self.vit.head.in_features
        self.vit.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# Load the model from the .pth file
def load_model(model_path, num_classes):
    model = ViT_StyleGAN_Model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model

# Load ViT model for audio
def load_vit_model(vit_path):
    model = ViTDeepfakeDetector()
    vit_checkpoint = torch.load(vit_path, map_location=torch.device('cpu'))
    model.load_state_dict(vit_checkpoint['model'] if 'model' in vit_checkpoint else vit_checkpoint, strict=False)
    model.eval()
    return model

# Preprocess video frames
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    frame = transform(frame)
    return frame

# Function to load and process video frames
def load_video_frames(video_path, max_frames=15):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame)
        frames.append(frame)
    cap.release()
    gc.collect()  # Clean up unused variables

    if len(frames) < max_frames:
        while len(frames) < max_frames:
            frames.append(torch.zeros((3, 224, 224)))
    else:
        frames = frames[:max_frames]

    frames = torch.stack(frames)
    return frames.unsqueeze(0)

# Predict video using StyleGAN2
def predict_video_with_stylegan2(model, video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    frames = load_video_frames(video_path).to(device)
    with torch.no_grad():
        outputs = model(frames)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Preprocess video with audio
def preprocess_video_with_audio(video_path, audio_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)  # Get timestamp in milliseconds
        timestamps.append(timestamp)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (224, 224))  # Reduce size
        frames.append(frame)

    cap.release()
    gc.collect()  # Clean up unused variables

    mel = get_spectrogram(audio_path)
    mel = cv2.resize(mel, (224 * len(frames) // 2, 224))  # Resize to half of original width to reduce memory usage

    mel = np.expand_dims(mel, axis=2)
    mel = np.repeat(mel, 3, axis=2)

    combined_images = []
    for i in range(0, len(frames), 5):  # Adjust window size if needed
        try:
            window_frames = frames[i:i+5]
            combined_frame = np.concatenate(window_frames, axis=1)

            start = int((i / len(frames)) * mel.shape[1])
            end = int(((i + 5) / len(frames)) * mel.shape[1])
            mel_section = mel[:, start:end]
            mel_section = cv2.resize(mel_section, (combined_frame.shape[1], combined_frame.shape[0]))

            combined_image = np.concatenate((mel_section, combined_frame), axis=0)
            combined_images.append((combined_image, timestamps[i]))
        except MemoryError:
            st.warning("Memory Error: Skipping frames due to large size.")
            break

    return combined_images

# Extract audio from video
def extract_audio_from_video(video_path, audio_output_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output_path, codec='pcm_s16le')
    except Exception as e:
        st.error(f"No audio for the video that you provided")

# Get spectrogram from audio file
def get_spectrogram(audio_file):
    data, sr = librosa.load(audio_file, sr=None)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=sr), ref=np.min)
    return mel

# Ensure directory exists
def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Streamlit UI
def main():
    st.title("Deepfake Detection Using Combined Models")

    # Load models
    model_pth_path = st.text_input('Path to the StyleGAN2 model (.pth file):', r'/Users/gowsiaravindk/Desktop/vit_stylegan_model.pth')
    vit_checkpoint_path = st.text_input('Path to the ViT model (.pth file for audio):', r'/Users/gowsiaravindk/Downloads/ckpt.pth')

    if not os.path.exists(model_pth_path):
        st.error("The specified StyleGAN2 model file does not exist.")
        return

    if not os.path.exists(vit_checkpoint_path):
        st.error("The specified ViT model file does not exist.")
        return

    # Load models
    stylegan_model = load_model(model_pth_path, num_classes=2)
    vit_model = load_vit_model(vit_checkpoint_path)

    # Streamlit file uploader
    uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "mov", "avi"])

    result = "Not determined"  # Initialize result variable

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            video_path = temp_file.name

        st.video(uploaded_file, format="video/mp4")

        with st.spinner("Processing video..."):
            # Determine if video has audio
            audio_path = tempfile.mktemp(suffix=".wav")
            extract_audio_from_video(video_path, audio_path)
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                combined_images = preprocess_video_with_audio(video_path, audio_path)

                preprocess_vit = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                vit_predictions = []
                frame_results = []
                for img, timestamp in combined_images:
                    img_tensor = preprocess_vit(img)
                    with torch.no_grad():
                        vit_output = vit_model(img_tensor.unsqueeze(0))
                        vit_probabilities = torch.softmax(vit_output, dim=1)
                        vit_fake_prob = vit_probabilities[0][1].item()
                        frame_results.append((timestamp, vit_fake_prob))
                        vit_predictions.append(vit_fake_prob)

                avg_vit_prediction = sum(vit_predictions) / len(vit_predictions)
                result = "Fake" if avg_vit_prediction > 0.5 else "Not Fake"

                st.write(f"Deepfake Detection Result: {result}")
                st.write("Frame Results (Timestamp: Probability):")
                for timestamp, probability in frame_results:
                    st.write(f"Time {timestamp:.2f} ms: Probability {probability:.2f}")
            else:
                prediction = predict_video_with_stylegan2(stylegan_model, video_path)
                result = "FAKE" if prediction == 1 else "REAL"
                st.write(f"Deepfake Detection Result: {result}")

if __name__ == "__main__":
    main()
