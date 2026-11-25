<p align="center">
  <img src="assets/logo.png" width="130" alt="CropCare AI logo">
</p>

<h1 align="center">CropCare AI</h1>

<p align="center">
  <b>Crop Disease Detection from Leaf Photos — with a Lightweight CNN and Offline Mobile Inference</b><br>
  Take a photo of a leaf. Get the disease name, confidence, treatment and prevention — in English or Urdu, with or without internet.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Flutter-3.x-02569B?logo=flutter&logoColor=white" alt="Flutter">
  <img src="https://img.shields.io/badge/TensorFlow%20Lite-INT8-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow Lite">
  <img src="https://img.shields.io/badge/Test%20Accuracy-95.40%25-success" alt="Test accuracy 95.40%">
  <img src="https://img.shields.io/badge/Works-Online%20%2B%20Offline-blue" alt="Online and offline">
  <img src="https://img.shields.io/badge/Languages-English%20%2B%20Urdu-green" alt="English and Urdu">
</p>

---

## About

**CropCare AI** is a BS Computer Science Final Year Project (FYP) at **Abdul Wali Khan University Mardan (AWKUM), Garden Campus**. It helps farmers detect crop diseases early: the user takes a photo of a leaf, and a Convolutional Neural Network (CNN) running **entirely on the phone** identifies the disease and shows treatment and prevention advice.

No internet is needed for diagnosis — an important requirement for farms and villages with weak or no signal. When internet is available, saved records can sync to a backend server.

## Features

- **On-device AI diagnosis** — INT8-quantized TensorFlow Lite model (3.53 MB) runs fully offline on the phone
- **12 disease classes** across Maize, Potato and Tomato (see list below)
- **Treatment & prevention advice** for every class, in **English and Urdu**
- **Login screen** (name + mobile, stored locally on the phone) with a **Guest mode**, so no farmer is ever blocked from a diagnosis
- **Personal profile** — user details, total diagnoses, pending-sync count and last diagnosis
- **Diagnosis history** saved locally in SQLite, with an offline sync queue
- **Online / Offline badge** on the home screen, plus a manual **Sync Now** option
- **AI Assistant (offline)** — built-in chat that answers questions about all 12 diseases, photo tips, fertilizer, watering and prevention, in English, Urdu or Roman Urdu
- **Voice Assistant** — ask questions by speaking (speech-to-text) and hear the answers spoken back (text-to-speech)
- **Honest low-confidence handling** — below 60% confidence the app asks for a clearer photo instead of giving a possibly wrong answer

## Model & Results

| Item | Detail |
|---|---|
| Architecture | MobileNetV3 (transfer learning), trained in Python / TensorFlow-Keras |
| Dataset | PlantVillage (held-out test set, duplicates removed) |
| Classes | 12 (Maize 3, Potato 3, Tomato 6) |
| Test accuracy | **95.40%** |
| Macro F1-score | **0.942** |
| Mobile model | `model.tflite` — INT8 quantized, **3.53 MB**, input 224×224 |
| Inference | 100% on-device (TensorFlow Lite), no server needed |

### Supported classes

| Crop | Classes |
|---|---|
| Maize (Corn) | Common Rust, Northern Leaf Blight, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Tomato Yellow Leaf Curl Virus, Healthy |

> **Planned expansion:** Wheat (5 classes) and Sugarcane (5 classes) — 22 classes in total — using additional Kaggle datasets, since these crops are not part of PlantVillage.

## Tech Stack

| Layer | Technology |
|---|---|
| Mobile app | Flutter / Dart |
| AI model | TensorFlow / Keras → TensorFlow Lite (INT8 quantization) |
| Training | Python, Google Colab / Kaggle notebooks |
| Local storage | SQLite (sqflite) |
| Voice | speech_to_text, flutter_tts |
| Backend (sync) | FastAPI + PostgreSQL (proposal design) |

## Getting Started

### Prerequisites
- [Flutter SDK](https://docs.flutter.dev/get-started/install) (3.x) and Android Studio with the Android SDK
- An Android phone, or a borrowed one for testing (USB debugging enabled)

### Run the app

```bash
# 1. Get this code (clone the repository or download the ZIP)
# 2. Create a Flutter project and copy these files into it:
#    lib/main.dart  ->  <project>/lib/main.dart
#    pubspec.yaml   ->  <project>/pubspec.yaml
#    assets/*       ->  <project>/assets/        (model.tflite, labels.txt, logo.png, plants_bg.jpg)
#    android/app/src/main/res/*  ->  app icon files (Logo: AI Scan Leaf)

# 3. In android/app/src/main/AndroidManifest.xml:
#    - set  android:label="CropCare AI"
#    - add the two permission lines from ANDROID_PERMISSIONS.txt (microphone + internet)

# 4. Install packages and build
flutter pub get
flutter build apk        # APK: build/app/outputs/flutter-apk/app-release.apk
# or run directly on a connected phone:
flutter run
```

## Project Structure

```
├── lib/
│   └── main.dart            # Complete app: UI, TFLite classifier, SQLite, sync, AI + voice assistant
├── assets/
│   ├── model.tflite         # INT8 quantized MobileNetV3 model (3.53 MB)
│   ├── labels.txt           # 12 class labels (exact training order)
│   ├── logo.png             # App logo (AI Scan Leaf)
│   └── plants_bg.jpg        # Home-screen background (bamboo)
├── android/app/src/main/res/  # Android app-icon files (all mipmap sizes)
├── pubspec.yaml             # Flutter dependencies
└── ANDROID_PERMISSIONS.txt  # Extra manifest steps for the voice assistant
```

## Team

| Member | Roll No. |
|---|---|
| Izaz Ahmad | 231005345 |
| Asif Malak | 231016232 |
| Muhammad Zaheer Uddin Babar | 231012246 |

**Supervisor:** Mam Kiran Falak
**Department:** Computer Science, Abdul Wali Khan University Mardan (AWKUM), Garden Campus
**Session:** 2026–2027

## Future Work

- Train the expanded **22-class** model (adding Wheat and Sugarcane)
- Field testing with real farm photos (outside PlantVillage lab conditions)
- Complete the FastAPI sync backend so records sync to a live server
- Publish the app for farmer use in the Mardan region

## License

This project is released under the [MIT License](LICENSE) — for academic and learning use.
