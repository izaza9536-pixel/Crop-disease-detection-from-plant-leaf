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
  <img src="https://img.shields.io/badge/TensorFlow%20Lite-Float16-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow Lite">
  <img src="https://img.shields.io/badge/Test%20Accuracy-94.06%25-success" alt="Test accuracy 94.06%">
  <img src="https://img.shields.io/badge/Works-Online%20%2B%20Offline-blue" alt="Online and offline">
  <img src="https://img.shields.io/badge/Languages-English%20%2B%20Urdu-green" alt="English and Urdu">
</p>

---

## About

**CropCare AI** is a personal project, designed and built by **Izaz Ahmad**. It helps farmers detect crop diseases early: the user takes a photo of a leaf, and a Convolutional Neural Network (CNN) running **entirely on the phone** identifies the disease and shows treatment and prevention advice.

No internet is needed for diagnosis — an important requirement for farms and villages with weak or no signal. When internet is available, saved records can sync to a backend server.

## Features

- **On-device AI diagnosis** — Float16 TensorFlow Lite model (6.11 MB) runs fully offline on the phone
- **32 disease classes** across **7 crops** — Maize, Potato, Tomato, Wheat, Sugarcane, Rice and Cotton (see list below)
- **Scanner-style camera with auto-capture** — live preview with a framing guide; the scanner takes the photo by itself when a leaf is properly in frame (manual shutter also available); camera photos are auto-straightened and centre-cropped before diagnosis, gallery photos work too
- **Safety gate (two-stage AI)** — a small 3-class gate model checks every photo first: non-leaf objects are refused with "This does not look like a leaf", and plants outside the 7 known crops get an honest "This plant is not in our system" instead of a forced wrong answer
- **Angle-robust diagnosis** — predictions are averaged over 4 rotations (0°/90°/180°/270°), so tilted leaf photos still diagnose correctly
- **Treatment & prevention advice** for every class, in **English and Urdu**
- **Login screen** (name + mobile, stored locally on the phone) with a **Guest mode**, so no farmer is ever blocked from a diagnosis
- **Personal profile** — user details, total diagnoses, pending-sync count and last diagnosis
- **Diagnosis history** saved locally in SQLite, with an offline sync queue
- **Online / Offline badge** on the home screen, plus a manual **Sync Now** option
- **AI Assistant (offline)** — built-in chat that answers questions about all 32 diseases, photo tips, fertilizer, watering and prevention, in English, Urdu or Roman Urdu
- **Voice Assistant** — ask questions by speaking (speech-to-text) and hear the answers spoken back (text-to-speech)
- **Honest low-confidence handling** — below 60% confidence the app asks for a clearer photo instead of giving a possibly wrong answer

## Model & Results

| Item | Detail |
|---|---|
| Architecture | MobileNetV3Large (transfer learning, two-stage fine-tuning), trained in Python / TensorFlow-Keras |
| Dataset | PlantVillage + Kaggle datasets for Wheat, Sugarcane, Rice and Cotton — ~32,300 images, duplicates removed (MD5), 80/10/10 split |
| Classes | 32 (Tomato 6, Rice 6, Wheat 5, Sugarcane 5, Cotton 4, Maize 3, Potato 3) |
| Test accuracy | **94.06%** (2,960 of 3,147 held-out test photos) |
| Macro F1-score | **0.926** |
| Mobile model | `model.tflite` — Float16 quantized, **6.11 MB**, input 224×224 |
| Gate model | `gate_model.tflite` — MobileNetV3Small, 3 classes (known leaf / other leaf / not a leaf), Float16, 1.95 MB |
| Gate test accuracy | **91.85%** (3,300 held-out photos); trained with PlantVillage + PlantDoc field photos + Caltech-256 real-world objects, so everyday objects (packets, hands, tools) are refused as "not a leaf" |
| Inference | 100% on-device (TensorFlow Lite), no server needed |

**Accuracy by crop (held-out test set):**

| Crop | Accuracy |
|---|---|
| Potato | 99.5% |
| Maize | 99.4% |
| Cotton | 98.1% |
| Tomato | 96.3% |
| Sugarcane | 89.9% |
| Wheat | 89.5% |
| Rice | 85.7% |

> **Quantization finding:** full-integer INT8 quantization was evaluated and **rejected** — its accuracy collapsed on this MobileNetV3 model (a known hard-swish sensitivity), so the app ships the Float16 model, which keeps full accuracy at 6.11 MB. A dynamic-range quantized version (3.34 MB) is kept as a smaller alternative.

### Supported classes

| Crop | Classes |
|---|---|
| Maize (Corn) | Common Rust, Northern Leaf Blight, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Tomato Yellow Leaf Curl Virus, Healthy |
| Wheat | Brown Rust, Powdery Mildew, Septoria Leaf Blotch, Yellow Rust, Healthy |
| Sugarcane | Mosaic, Red Rot, Rust, Yellow Leaf, Healthy |
| Rice | Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Sheath Blight, Healthy |
| Cotton | Bacterial Blight, Fusarium Wilt, Leaf Curl Virus, Healthy |

## Tech Stack

| Layer | Technology |
|---|---|
| Mobile app | Flutter / Dart |
| AI model | TensorFlow / Keras → TensorFlow Lite (Float16 quantization) |
| Training | Python, Kaggle notebooks (GPU) |
| Local storage | SQLite (sqflite) |
| Voice | speech_to_text, flutter_tts |
| Backend (sync) | FastAPI + PostgreSQL (planned) |

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
#    assets/*       ->  <project>/assets/        (model.tflite, labels.txt, gate_model.tflite, gate_labels.txt, logo.png, plants_bg.jpg)
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
│   ├── model.tflite         # Float16 MobileNetV3 model (6.11 MB)
│   ├── labels.txt           # 32 class labels (exact training order)
│   ├── gate_model.tflite    # Float16 gate model (1.95 MB) — known leaf / other plant / not a leaf
│   ├── gate_labels.txt      # 3 gate class labels
│   ├── logo.png             # App logo (AI Scan Leaf)
│   └── plants_bg.jpg        # Home-screen background (bamboo)
├── android/app/src/main/res/  # Android app-icon files (all mipmap sizes)
├── pubspec.yaml             # Flutter dependencies
└── ANDROID_PERMISSIONS.txt  # Extra manifest steps for the voice assistant
```

## Future Work

- Field testing with real farm photos (outside PlantVillage lab conditions) and fine-tuning on field data
- Grad-CAM visual explanations for each diagnosis
- Complete the FastAPI sync backend so records sync to a live server
- Publish the app for farmer use in the Mardan region

## License

This project is released under the [MIT License](LICENSE) — for learning and personal use.
