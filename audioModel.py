import tensorflow as tf
import librosa
import numpy as np
import cv2

# Define constants
SAMPLE_RATE = 16000  # Sample rate for loading audio (in Hz)
DURATION = 3  # Duration of audio clips in seconds
N_MELS = 128  # Number of Mel bands
MAX_TIME_STEPS = 109  # Ensure all spectrograms have the same time dimension
N_BINS = 84          # Number of frequency bins for CQT
HOP_LENGTH = 512     # Hop length for CQT
INPUT_SHAPE = (224, 224, 3)  # Expected input shape for the model (resized)

# Load the single pre-trained model
audio_model = tf.keras.models.load_model('models/audio/audio_classifier.h5')
audio_model2 = tf.keras.models.load_model('models/audio/audio_classifier_2.h5')

def preprocess_audio(file_path):
    """
    Preprocesses an audio file for model prediction.
    - Loads the audio, converts to Mel spectrogram, normalizes, and reshapes for the model.
    """
    # Load audio file
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Extract Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    
    # converts spectrogram's power values to decibel scale, making it more interpretable for model
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max) 

    # Ensure fixed width in time dimension
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    # Expand dimensions for model input (height, width, channels)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # Add channel dimension
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension

    return mel_spectrogram

def preprocess_audio2(file_path, input_shape=(128, 128)):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Compute CQT spectrogram
    cqt = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)

    # Resize to 128x128 (as expected by the model)
    cqt_resized = cv2.resize(cqt, input_shape, interpolation=cv2.INTER_AREA)

    # Normalize the CQT spectrogram
    cqt_resized = cqt_resized / np.max(cqt_resized)

    # Repeat the single channel to create 3 channels (to match the model input)
    cqt_resized = np.stack([cqt_resized] * 3, axis=-1)

    return cqt_resized  # Shape: (128, 128, 3)

def predict_audio2(file_path):
        """
        Make a prediction on the given audio file.
        :return: Predicted class, probability of 'fake', probability of 'real'.
        """
        try:
            # Preprocess audio
            audio_features = preprocess_audio2(file_path)
            audio_features = np.expand_dims(audio_features, axis=0)  # Add batch dimension
            
            # Predict
            predictions = audio_model2.predict(audio_features)
            print(predictions)
            score_fake = predictions[0][0]  # Probability of 'real'
            score_real = predictions[0][1]  # Probability of 'fake'
            
            # Determine class
            predicted_class = 'real' if score_real > score_fake else 'fake'
            
            return predicted_class, score_fake, score_real
        
        except Exception as e:
            raise ValueError(f"Error in prediction: {e}")

def predict_audio(file_path):
    """
    Predicts if an audio file is bonafide or spoof using the trained model.
    """
    # Preprocess the audio file
    mel_spectrogram = preprocess_audio(file_path)

    # Get prediction
    prediction = audio_model.predict(mel_spectrogram)[0][0]
    print(prediction)

    # Determine final class based on the prediction score
    predicted_class = 'Real' if prediction > 0.5 else 'Fake'
    score_real = round(float(prediction * 100), 2)
    score_fake = round(100 - score_real, 2)

    return predicted_class, score_real, score_fake
