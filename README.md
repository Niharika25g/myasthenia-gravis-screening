# Multimodal Myasthenia Gravis (MG) Screening Framework

A lightweight, web-based preliminary screening tool for **Myasthenia Gravis (MG)** that combines computer vision and acoustic speech analysis into a unified Multi-Layer Perceptron (MLP) classifier.

## Overview
Myasthenia Gravis frequently affects ocular and bulbar muscles, causing eyelid drooping (ptosis) and vocal fatigue. This application extracts scale-invariant facial landmark metrics and speech features from user-uploaded video or audio recordings, fuses them into a **30-dimensional feature vector**, and outputs an explainable clinical risk report backed by an SQLite audit log.

## Key Features
- **Multimodal Feature Extraction:**
  - **Visual Pipeline (`visual_module.py`):** Uses OpenCV and MediaPipe Face Mesh to compute 15 normalized ocular ptosis statistics (mean, standard deviation, minimum, skewness, and fatigue proportion).
  - **Acoustic Pipeline (`speech_module.py`):** Uses MoviePy and Librosa (16 kHz) to extract 15 vocal markers (13 MFCCs, RMS energy, and spectral centroid).
- **Feature-Level Fusion & Classification (`multimodal_fusion.py`, `train_model.py`):** Concatenates the 15 facial and 15 acoustic metrics into a 30-feature input standardized via `StandardScaler` and evaluated by an MLP classifier (`(32, 16)` hidden layers, ReLU activation, Adam optimizer), achieving **93.75% test accuracy** and **0.99 ROC-AUC**.
- **Web Interface & Persistence (`app.py`, `database.py`):** Flask web application supporting video (`.mp4`, `.avi`) and audio-only (`.wav`, `.mp3` with zero-padded facial features) uploads, instant biomarker breakdowns, model evaluation views, and persistent SQLite screening history.

## Tech Stack
- **Backend & Web:** Python, Flask, Jinja2
- **Computer Vision:** OpenCV, MediaPipe
- **Audio & Signal Processing:** Librosa, MoviePy
- **Machine Learning:** Scikit-Learn, NumPy, Joblib
- **Database:** SQLite

## Project Structure
```text
MG_Project/
├── app.py                    # Main Flask server, routes, and inference orchestration
├── train_model.py            # Offline dataset preprocessing, scaling, and MLP training
├── visual_module.py          # OpenCV + MediaPipe facial ptosis feature extraction
├── speech_module.py          # MoviePy + Librosa acoustic feature extraction
├── multimodal_fusion.py      # Feature-level fusion (15 facial + 15 acoustic -> 30D)
├── database.py               # SQLite database initialization and audit logging
├── symptom_agent.py          # Risk interpretation and clinical recommendation logic
├── templates/                # HTML views (index, result, history, evaluation)
├── static/                   # CSS and frontend assets
└── uploads/                  # Temporary upload directory