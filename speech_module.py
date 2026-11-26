import librosa
import numpy as np
import os

def extract_speech_features(audio_path):
    y, sr = librosa.load(audio_path, duration=30.0)
    
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    pitch, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    mean_pitch = np.nanmean(pitch)
    rms_energy = np.mean(librosa.feature.rms(y=y))
    
    return {
        "mfcc_vector": mfccs.tolist(),
        "mean_pitch": float(mean_pitch) if not np.isnan(mean_pitch) else 0.0,
        "energy_level": float(rms_energy)
    }

def evaluate_audio_dataset(folder_path="dataset_audio"):
    summary = []
    if not os.path.exists(folder_path):
        return [{"file": "Folder missing", "error": True}]
        
    for filename in os.listdir(folder_path):
        if filename.endswith(('.wav', '.mp3', '.flac')):
            path = os.path.join(folder_path, filename)
            features = extract_speech_features(path)
            summary.append({"file": filename, "features": features})
            
    return summary