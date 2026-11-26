import os
import numpy as np
import joblib
from database import db, ScreeningRecord, init_db
from flask import Flask
from visual_module import process_facial_video
from speech_module import extract_speech_features
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
init_db(app)

def process_and_store_datasets():
    with app.app_context():
        print("[INFO] Processing dataset files and logging to database...")
        X, y = [], []
        
        # 1. Process Video Folder
        video_folder = "dataset_videos"
        if os.path.exists(video_folder):
            files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
            print(f"[INFO] Found {len(files)} video(s) to process.")
            for filename in files:
                print(f" -> Processing video: {filename}...")
                path = os.path.join(video_folder, filename)
                metrics = process_facial_video(path)
                
                # Uniform feature vector of length 15
                v_feat = [metrics["ptosis_ratio"], metrics["frames_processed"] / 1000.0]
                feature_vector = v_feat + [0.0] * 13
                X.append(feature_vector)
                
                label = 1 if metrics["ptosis_ratio"] > 0.05 else 0
                y.append(label)
                
                # Log to database
                record = ScreeningRecord(
                    filename=filename,
                    modality="Video",
                    risk_level="High Risk" if label == 1 else "Low Risk",
                    probability=round(metrics["ptosis_ratio"] * 10, 2)
                )
                db.session.add(record)

        # 2. Process Audio Folder
        audio_folder = "dataset_audio"
        if os.path.exists(audio_folder):
            audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3', '.flac'))]
            print(f"[INFO] Found {len(audio_files)} audio file(s) to process.")
            for filename in audio_files:
                print(f" -> Processing audio: {filename}...")
                path = os.path.join(audio_folder, filename)
                feats = extract_speech_features(path)
                
                # Ensure audio feature vector is also padded/trimmed to length 15
                s_feat = feats["mfcc_vector"] + [feats["mean_pitch"] / 500.0, feats["energy_level"]]
                if len(s_feat) < 15:
                    s_feat += [0.0] * (15 - len(s_feat))
                else:
                    s_feat = s_feat[:15]
                    
                X.append(s_feat)
                label = 1 if feats["energy_level"] < 0.02 else 0
                y.append(label)
                
                # Log to database
                record = ScreeningRecord(
                    filename=filename,
                    modality="Audio",
                    risk_level="High Risk" if label == 1 else "Low Risk",
                    probability=round(1.0 - feats["energy_level"] * 5, 2)
                )
                db.session.add(record)
                    
        db.session.commit()
        print("[INFO] All dataset files successfully processed and logged to SQLite database!")

        # Train Neural Network Model
        if len(X) > 0:
            X, y = np.array(X), np.array(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, "mg_ann_model.pkl")
            print("[INFO] Neural Network model trained and saved as 'mg_ann_model.pkl'!")

if __name__ == "__main__":
    process_and_store_datasets()