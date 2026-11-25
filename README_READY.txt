CropCare AI - FINAL updated package (2026-11-25)

WHAT IS NEW:
1. Name: CropCare AI
0. BACKGROUND is now BAMBOO TREE (assets/plants_bg.jpg - same file name, new picture)
0b. PROFILE: person icon on home top bar - user name, mobile, total diagnoses, pending sync, last diagnosis, History + Logout buttons
0d. ONLINE + OFFLINE: home screen par Online/Offline badge hai (dabane se dobara check). Offline main diagnosis + assistant + history chalta hai; online main record khud sync ki koshish karte hain, Profile main 'Sync Now' button bhi hai.
0c. AI ASSISTANT now answers in English AND Urdu automatically (English / Urdu / Roman Urdu sawal - same language main jawab) and has VOICE: mic button se bol kar sawal, jawab awaz main bhi sunata hai (speaker on/off upar hai). Voice ke liye ANDROID_PERMISSIONS.txt wala aik step lazmi hai + flutter pub get (2 naye packages: speech_to_text, flutter_tts)
2. Logo: Logo 2 (AI Scan Leaf) - app icon files are in android/app/src/main/res (all mipmap sizes) + shown inside the app
3. Background: soft plants + wheat picture (assets/plants_bg.jpg) behind the home screen
4. Urdu: tap UrdU button (top right) - buttons, result, disease names (all 12), treatment, prevention and History all switch to Urdu
5. Login: a DIFFERENT dark-green login interface opens first - name + mobile number, saved on the phone only (offline). 'Continue as Guest' also works. Logout button is on the home screen top bar.
6. AI Assistant: home screen ke neeche AI Assistant button hai - offline chat, English/Urdu, saari 12 bimariyon ka ilaaj/ah­tiyaat, photo tips, khad/pani ke jawabat. Internet ya API key ki zaroorat nahi.
7. Green top bar - the Urdu button is now clearly visible (it was white on white before).

COPY STEPS (same as before):
- lib/main.dart  -> D:\crop_disease_app\lib\main.dart (replace)
- pubspec.yaml   -> D:\crop_disease_app\pubspec.yaml (replace)
- assets folder ke saari files -> D:\crop_disease_app\assets\ (model.tflite, labels.txt, plants_bg.jpg, logo.png)
- android folder -> D:\crop_disease_app\ ke andar merge karo (sirf icon files hain, replace kar do)
- AndroidManifest.xml main android:label="CropCare AI" kar do (phone icon ke neeche naam ke liye)
Phir cmd main:
  cd /d D:\crop_disease_app
  flutter pub get
  flutter build apk
