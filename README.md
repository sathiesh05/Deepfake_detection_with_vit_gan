# Deepfake Detection Using Combined Models

This project implements a deepfake detection system utilizing a combination of Vision Transformer (ViT) and StyleGAN models. The system analyzes video frames and audio to classify content as either real or fake.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Video Frame Analysis:** Extracts frames from uploaded videos and analyzes them using a pre-trained ViT model.
- **Audio Processing:** Extracts audio from videos and computes a spectrogram for further analysis.
- **Deepfake Detection:** Combines visual and audio data to determine the authenticity of the video content.
- **Streamlit Interface:** Provides a user-friendly interface for uploading videos and viewing results.

## Requirements

- Python 3.7+
- Libraries:
  - PyTorch
  - torchvision
  - transformers
  - streamlit
  - librosa
  - opencv-python
  - moviepy
  - timm

You can install the required libraries using:

```bash
pip install torch torchvision transformers streamlit librosa opencv-python moviepy timm
```

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/deepfake-detection.git
    cd deepfake-detection
    ```

2. Install the required packages (as mentioned above).

3. Download the pre-trained models (StyleGAN and ViT) and save them in a known directory.

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload a video file (supported formats: MP4, MOV, AVI).

4. Enter the paths to the StyleGAN and ViT model `.pth` files.

5. Click on "Process Video" to see the deepfake detection results.

## Model Details

### ViT_StyleGAN_Model

This model combines the Vision Transformer architecture with a StyleGAN generator to extract and classify features from video frames.

### ViTDeepfakeDetector

This model uses a ViT architecture to analyze audio spectrograms for deepfake detection.

## Contributing

Contributions are welcome! If you find any issues or want to add features, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
