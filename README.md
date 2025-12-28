# Emotion Detection App 🎭

A modular, real-time emotion detection system built with **TensorFlow/Keras**, **OpenCV**, and **Streamlit**.  
This project supports image uploads and live webcam streaming, with emotion classification powered by a CNN trained on the FER-2013 dataset.

---

## 🚀 Features
- Upload images for emotion detection.
- Stream webcam video with live emotion prediction.
- Modular pipeline: preprocessing, training, evaluation, and inference.
- Clean Streamlit UI for interaction.
- Ready for deployment on Streamlit Cloud.

---

## 📁 Project Structure
```
EMOTION_DETECTION/
│
├── archive/                  # Raw FER-2013 dataset (train/test folders)
│
├── data/                     # Preprocessed NumPy arrays
│   ├── train_x.npy
│   ├── train_y.npy
│   ├── val_x.npy
│   ├── val_y.npy
│   ├── test_x.npy
│   └── test_y.npy
│
├── Images/                   # Sample images for testing
│
├── models/                   # Saved model files
│   └── emotion_model.h5
│
├── scripts/
│   ├── models/               # Jupyter notebooks for each stage
│   │   ├── download_fer2013.ipynb
│   │   ├── preprocess_fer2013.ipynb
│   │   ├── train.ipynb
│   │   ├── evaluate.ipynb
│   │   └── inference.ipynb
│   └── app.py                # Streamlit app entry point
│
├── venv/                     # Virtual environment
│
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **TensorFlow / Keras** – CNN model for emotion classification
- **OpenCV** – face detection and image processing
- **Streamlit** – interactive web UI
- **NumPy & Pillow** – image handling

---

## ⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/nithin8688/emotion-detection-app.git
   cd emotion-detection-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate   # Windows PowerShell
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Run the App

```bash
python -m streamlit run scripts/app.py
```

- **Upload Mode**: Drag and drop an image for emotion prediction.
- **Webcam Mode**: Click **Start Camera** to stream live video.  
  - Stop with **Stop Camera** button or press `q/e/z`.

---

## 📊 Model Details
- CNN trained on **FER-2013 dataset**
- Input shape: `(48, 48, 1)` grayscale
- Output classes: `Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`

---

## 🌐 Deploy to Streamlit Cloud

1. Push your project to GitHub (including `app.py`, `emotion_model.h5`, and `requirements.txt`)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo and select `scripts/app.py`
4. Share your public app link!

---

## 🤝 Contributing
Pull requests are welcome. For major changes, open an issue first to discuss what you’d like to change.

---

## 📜 License
This project is licensed under the MIT License.
```
