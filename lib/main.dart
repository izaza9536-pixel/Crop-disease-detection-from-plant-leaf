/// CropCare AI - Crop Disease Detection - Flutter app (Proposal Sections 6.2 and 6.3).
///
/// Updated 2026-11-25 for Izaz's trained 32-class model (7 crops):
///   * Works with the FLOAT16 model.tflite from the notebook (6.11 MB,
///     lab accuracy 94.06%). Its input is FLOAT raw pixels (0-255) -
///     the [-1, 1] scaling lives INSIDE the model, so the app must
///     NOT scale. (Full INT8 quantization was tested and rejected:
///     its accuracy collapsed - see the project verification notes.)
///   * labels.txt = the 32 classes of the 2026-11-25 run, in the exact
///     order the notebook produced (alphabetical). Never edit by hand.
///   * Treatment advice filled in for all 32 classes (English + Urdu).
///
/// What the app does:
///   * Take a leaf photo (camera) or pick one from the gallery.
///   * Run the TFLite CNN ON THE PHONE (fully offline) - no internet needed.
///   * Show disease name + confidence. If confidence is low, it honestly
///     says "uncertain - take a clearer photo" (proposal Section 8.3).
///   * Show treatment / prevention advice in English and Urdu.
///   * Save every diagnosis in a local SQLite history (sqflite).
///   * Sync: records are marked synced=0; when internet returns, the app
///     uploads the queue to the FastAPI backend (see backend/main.py).
///
/// Files you must add:
///   assets/model.tflite   <- from the notebook results (Step 8/9)
///   assets/labels.txt     <- included in this package (32 classes)
/// Packages (see pubspec.yaml): image_picker, tflite_flutter, image,
/// sqflite, path, http.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:camera/camera.dart' as cam;

// ---------------------------------------------------------------------------
// CONFIG - change these for your project
// ---------------------------------------------------------------------------
const String kModelAsset = 'assets/model.tflite';
const String kLabelsAsset = 'assets/labels.txt';
const int kInputSize = 224;                 // MobileNetV3 input size
const double kConfidenceThreshold = 0.60;   // below this -> "uncertain" (8.3)
const String kBackendBaseUrl = 'http://10.0.2.2:8000'; // emulator -> PC
// For a real phone on the same Wi-Fi use your PC's IP, e.g. 192.168.1.10:8000
const int kModelVersion = 2;

// ---------------------------------------------------------------------------
// TREATMENT ADVICE (Section 6.2) - English + Urdu, all 32 classes.
// Key = the class label from labels.txt (exact same spelling).
// General guidance only - always follow the product label and local
// agriculture department advice for doses.
// ---------------------------------------------------------------------------
class Advice {
  final String treatmentEn, preventionEn, treatmentUr, preventionUr;
  const Advice(this.treatmentEn, this.preventionEn, this.treatmentUr, this.preventionUr);
}

const Map<String, Advice> kAdvice = {
  'Corn_(maize)___Common_rust_': Advice(
    'Apply a fungicide containing mancozeb or a triazole, as per label '
        'instructions. Remove badly infected leaves.',
    'Use resistant varieties, keep proper plant spacing, and avoid '
        'overhead irrigation.',
    'مینکوزیب یا ٹرائیازول والی فنگس مار دوا ہدایات کے مطابق استعمال کریں۔ '
        'زیادہ متاثرہ پتے ہٹا دیں۔',
    'مزاحم اقسام استعمال کریں، پودوں میں مناسب فاصلہ رکھیں اور اوپر سے '
        'آبپاشی سے پرہیز کریں۔',
  ),
  'Corn_(maize)___Northern_Leaf_Blight': Advice(
    'Apply a fungicide (for example azoxystrobin or a triazole) early, as '
        'per label instructions. Destroy infected crop residue after harvest.',
    'Use resistant hybrids, rotate crops, and bury or remove old maize '
        'residue from the field.',
    'ابتداء میں ہی فنگس مار دوا (مثلاً ایزوکسی اسٹروبن یا ٹرائیازول) ہدایات '
        'کے مطابق استعمال کریں۔ فصل کے بعد متاثرہ باقیات تلف کریں۔',
    'مزاحم ہائبرڈ استعمال کریں، فصلیں تبدیل کریں اور پرانی مکئی کی باقیات '
        'کھیت سے ہٹا دیں۔',
  ),
  'Corn_(maize)___healthy': Advice(
    'No treatment needed. Your maize plant looks healthy.',
    'Keep normal watering and check the leaves every week.',
    'کسی علاج کی ضرورت نہیں۔ آپ کی مکئی صحت مند لگ رہی ہے۔',
    'معمول کے مطابق پانی دیں اور ہر ہفتے پتے چیک کریں۔',
  ),
  'Potato___Early_blight': Advice(
    'Apply mancozeb or chlorothalonil fungicide as per label instructions. '
        'Remove and destroy infected leaves.',
    'Rotate crops, keep spacing for air flow, use balanced fertilizer, '
        'and avoid wetting the leaves while watering.',
    'مینکوزیب یا کلوروتھالونل فنگس مار دوا ہدایات کے مطابق استعمال کریں۔ '
        'متاثرہ پتے ہٹا کر تلف کریں۔',
    'فصلیں تبدیل کریں، ہوا کے لیے فاصلہ رکھیں، متوازن کھاد دیں اور پانی '
        'دیتے وقت پتے گیلے نہ کریں۔',
  ),
  'Potato___Late_blight': Advice(
    'Act fast - this disease spreads quickly. Apply a copper-based or '
        'metalaxyl-based fungicide as per label instructions and destroy '
        'badly infected plants.',
    'Use certified disease-free seed, keep spacing between plants, avoid '
        'overhead watering, and watch closely in cool, wet weather.',
    'جلدی کریں — یہ بیماری تیزی سے پھیلتی ہے۔ تانبے والی یا میٹالیکسل والی '
        'فنگس مار دوا ہدایات کے مطابق استعمال کریں اور زیادہ متاثرہ پودے تلف کریں۔',
    'تصدیق شدہ بیماری سے پاک بیج استعمال کریں، پودوں میں فاصلہ رکھیں، اوپر '
        'سے پانی نہ دیں اور ٹھنڈے گیلے موسم میں خاص نظر رکھیں۔',
  ),
  'Potato___healthy': Advice(
    'No treatment needed. Your potato plant looks healthy.',
    'Keep normal watering and check the leaves every week.',
    'کسی علاج کی ضرورت نہیں۔ آپ کا آلو صحت مند لگ رہا ہے۔',
    'معمول کے مطابق پانی دیں اور ہر ہفتے پتے چیک کریں۔',
  ),
  'Tomato___Bacterial_spot': Advice(
    'Apply a copper-based bactericide as per label instructions. Remove '
        'infected leaves and do not touch wet plants - the bacteria spread '
        'by hands and tools.',
    'Use disease-free seed, rotate crops, avoid overhead watering, and '
        'clean tools after use.',
    'تانبے والی بیکٹیریا مار دوا ہدایات کے مطابق استعمال کریں۔ متاثرہ پتے '
        'ہٹائیں اور گیلے پودوں کو ہاتھ نہ لگائیں — بیکٹیریا ہاتھوں اور اوزاروں '
        'سے پھیلتے ہیں۔',
    'بیماری سے پاک بیج استعمال کریں، فصلیں تبدیل کریں، اوپر سے پانی نہ دیں '
        'اور اوزار صاف رکھیں۔',
  ),
  'Tomato___Early_blight': Advice(
    'Remove infected lower leaves. Apply mancozeb or chlorothalonil '
        'fungicide as per label instructions. Put mulch under the plants '
        'so soil does not splash on the leaves.',
    'Keep spacing and stake the plants, water at the base (not on the '
        'leaves), and rotate crops each season.',
    'متاثرہ نچلے پتے ہٹائیں۔ مینکوزیب یا کلوروتھالونل فنگس مار دوا ہدایات '
        'کے مطابق استعمال کریں۔ پودوں کے نیچے ملچ ڈالیں تاکہ مٹی پتوں پر نہ اڑے۔',
    'پودوں میں فاصلہ رکھیں اور سہارا دیں، پانی جڑوں میں دیں (پتوں پر نہیں) '
        'اور ہر موسم میں فصل تبدیل کریں۔',
  ),
  'Tomato___Late_blight': Advice(
    'Remove and destroy infected leaves. Apply a copper-based fungicide or '
        'mancozeb as per label instructions. Avoid overhead watering.',
    'Use disease-free seed, keep space between plants for air flow, and '
        'rotate crops each season.',
    'متاثرہ پتے ہٹا کر تلف کریں۔ ہدایات کے مطابق تانبے والی فنگس مار دوا یا '
        'مینکوزیب استعمال کریں۔ اوپر سے پانی دینے سے پرہیز کریں۔',
    'بیماری سے پاک بیج استعمال کریں، پودوں کے درمیان ہوا کے لیے فاصلہ رکھیں '
        'اور ہر موسم میں فصل تبدیل کریں۔',
  ),
  'Tomato___Leaf_Mold': Advice(
    'Improve air flow around the plants and reduce humidity. Apply '
        'chlorothalonil or mancozeb fungicide as per label instructions.',
    'Keep spacing, prune lower leaves, water at the base, and avoid '
        'wetting the leaves.',
    'پودوں کے اردگرد ہوا کا گزر بہتر کریں اور نمی کم کریں۔ کلوروتھالونل یا '
        'مینکوزیب فنگس مار دوا ہدایات کے مطابق استعمال کریں۔',
    'پودوں میں فاصلہ رکھیں، نچلے پتے کاٹیں، پانی جڑوں میں دیں اور پتے '
        'گیلے نہ کریں۔',
  ),
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus': Advice(
    'There is no cure for this virus. Remove and destroy infected plants '
        'so it does not spread. Control whiteflies with yellow sticky traps '
        'and, if needed, an insecticide as per label instructions.',
    'Use resistant varieties, cover young plants with net, and keep the '
        'field free of weeds that host whiteflies.',
    'اس وائرس کا کوئی علاج نہیں۔ متاثرہ پودے ہٹا کر تلف کریں تاکہ بیماری '
        'نہ پھیلے۔ سفید مکھی کو پیلی چپکنے والی پٹیوں اور ضرورت پڑے تو ہدایات '
        'کے مطابق کیڑے مار دوا سے قابو کریں۔',
    'مزاحم اقسام استعمال کریں، چھوٹے پودوں کو جالی سے ڈھانپیں اور کھیت کو '
        'جڑی بوٹیوں سے صاف رکھیں۔',
  ),
  'Tomato___healthy': Advice(
    'No treatment needed. Your plant looks healthy.',
    'Keep normal watering and check leaves every week.',
    'کسی علاج کی ضرورت نہیں۔ آپ کا پودا صحت مند لگ رہا ہے۔',
    'معمول کے مطابق پانی دیں اور ہر ہفتے پتے چیک کریں۔',
  ),
  'Wheat___healthy': Advice(
    'No treatment needed. Your wheat looks healthy.',
    'Keep normal watering and check the leaves every week, especially '
        'in cool, humid weather when rusts spread.',
    'کسی علاج کی ضرورت نہیں۔ آپ کی گندم صحت مند لگ رہی ہے۔',
    'معمول کے مطابق پانی دیں اور ہر ہفتے پتے چیک کریں، خاص طور پر ٹھنڈے '
        'مرطوب موسم میں جب زنگ پھیلتا ہے۔',
  ),
  'Wheat___Yellow_rust': Advice(
    'Apply a triazole fungicide (for example propiconazole or '
        'tebuconazole) as per label instructions, as soon as you see '
        'the first yellow stripes. This rust spreads very fast in cool '
        'weather, so do not delay.',
    'Grow resistant varieties, sow on time, avoid too much nitrogen, '
        'and check the crop twice a week in February-March.',
    'جیسے ہی پہلی پیلی دھاریاں نظر آئیں، ٹرائیازول فنگس مار دوا (مثلاً '
        'پروپیکونازول یا ٹیبوکونازول) ہدایات کے مطابق فوراً استعمال کریں۔ '
        'یہ زنگ ٹھنڈے موسم میں بہت تیزی سے پھیلتا ہے، دیر نہ کریں۔',
    'مزاحم اقسام اگائیں، وقت پر بوائی کریں، نائٹروجن زیادہ نہ دیں اور '
        'فروری-مارچ میں ہفتے میں دو بار فصل چیک کریں۔',
  ),
  'Wheat___Brown_rust': Advice(
    'Apply a triazole fungicide (for example propiconazole) as per '
        'label instructions when pustules appear on the leaves.',
    'Grow resistant varieties, rotate crops, and remove volunteer '
        'wheat plants near the field.',
    'جب پتوں پر دانے نظر آئیں تو ٹرائیازول فنگس مار دوا (مثلاً '
        'پروپیکونازول) ہدایات کے مطابق استعمال کریں۔',
    'مزاحم اقسام اگائیں، فصلیں تبدیل کریں اور کھیت کے قریب خود رو گندم '
        'کے پودے ختم کریں۔',
  ),
  'Wheat___Powdery_mildew': Advice(
    'Apply wettable sulfur or a triazole fungicide as per label '
        'instructions. Remove badly infected lower leaves.',
    'Keep proper spacing for air flow, use balanced nitrogen (too '
        'much nitrogen makes it worse), and grow resistant varieties.',
    'سلفر یا ٹرائیازول فنگس مار دوا ہدایات کے مطابق استعمال کریں۔ '
        'زیادہ متاثرہ نچلے پتے ہٹا دیں۔',
    'ہوا کے گزر کے لیے مناسب فاصلہ رکھیں، متوازن نائٹروجن دیں (زیادہ '
        'نائٹروجن بیماری بڑھا دیتی ہے) اور مزاحم اقسام اگائیں۔',
  ),
  'Wheat___Septoria_leaf_blotch': Advice(
    'Apply chlorothalonil or a triazole fungicide as per label '
        'instructions. Protect the top (flag) leaf - it makes most of '
        'the grain.',
    'Rotate crops, bury or remove wheat stubble after harvest, grow '
        'resistant varieties, and avoid overhead irrigation.',
    'کلوروتھالونل یا ٹرائیازول فنگس مار دوا ہدایات کے مطابق استعمال '
        'کریں۔ سب سے اوپر والے پتے (فلیگ لیف) کی حفاظت کریں — دانہ زیادہ '
        'تر اسی سے بنتا ہے۔',
    'فصلیں تبدیل کریں، کٹائی کے بعد گندم کی باقیات دفن کریں یا ہٹائیں، '
        'مزاحم اقسام اگائیں اور اوپر سے آبپاشی نہ کریں۔',
  ),
  'Sugarcane___healthy': Advice(
    'No treatment needed. Your sugarcane looks healthy.',
    'Use healthy seed setts, keep normal watering, and check the '
        'leaves every week.',
    'کسی علاج کی ضرورت نہیں۔ آپ کا گنا صحت مند لگ رہا ہے۔',
    'صحت مند بیج (سم) استعمال کریں، معمول کے مطابق پانی دیں اور ہر '
        'ہفتے پتے چیک کریں۔',
  ),
  'Sugarcane___Red_rot': Advice(
    'There is no effective spray for red rot. Dig out and burn or '
        'bury the infected stools at once, and never take seed setts '
        'from an infected field.',
    'Plant only disease-free setts of resistant varieties, treat setts '
        'with fungicide or hot water before planting, rotate the field, '
        'and keep it drained - the fungus spreads through water.',
    'سرخ سڑن کا کوئی مؤثر سپرے نہیں۔ متاثرہ جڑیں (ٹھونٹھ) فوراً نکال '
        'کر جلا دیں یا دفن کریں، اور متاثرہ کھیت سے کبھی بیج (سم) نہ لیں۔',
    'صرف مزاحم اقسام کے بیماری سے پاک سم لگائیں، بوائی سے پہلے سم کو '
        'فنگس مار دوا یا گرم پانی سے علاج کریں، کھیت تبدیل کریں اور نکاسی '
        'اچھی رکھیں — یہ فنگس پانی سے پھیلتی ہے۔',
  ),
  'Sugarcane___Rust': Advice(
    'Apply mancozeb or a triazole fungicide as per label instructions '
        'when orange-brown pustules appear under the leaves.',
    'Grow resistant varieties, keep proper spacing, and use balanced '
        'fertilizer.',
    'جب پتوں کے نیچے نارنجی بھورے دانے نظر آئیں تو مینکوزیب یا '
        'ٹرائیازول فنگس مار دوا ہدایات کے مطابق استعمال کریں۔',
    'مزاحم اقسام اگائیں، مناسب فاصلہ رکھیں اور متوازن کھاد دیں۔',
  ),
  'Sugarcane___Mosaic': Advice(
    'This is a virus - there is no cure. Remove and destroy infected '
        'plants, and control aphids (the insects that carry it) with an '
        'insecticide as per label instructions if needed.',
    'Plant disease-free setts of resistant varieties, and keep the '
        'field and its borders free of weeds and grasses.',
    'یہ وائرس ہے — اس کا کوئی علاج نہیں۔ متاثرہ پودے ہٹا کر تلف کریں '
        'اور تیلے (اس کو پھیلانے والے کیڑے) کو ضرورت پڑے تو ہدایات کے '
        'مطابق کیڑے مار دوا سے قابو کریں۔',
    'مزاحم اقسام کے بیماری سے پاک سم لگائیں اور کھیت اور اس کے کناروں '
        'کو جڑی بوٹیوں اور گھاس سے صاف رکھیں۔',
  ),
  'Sugarcane___Yellow_leaf': Advice(
    'This yellowing is caused by a virus spread by aphids - there is '
        'no cure in the standing crop. Remove badly infected stools and '
        'control aphids as per label instructions.',
    'Always plant certified disease-free (tissue-culture) seed cane, '
        'and control aphids early in the season.',
    'یہ پیلاہٹ تیلے سے پھیلنے والے وائرس کی وجہ سے ہے — کھڑی فصل میں '
        'اس کا کوئی علاج نہیں۔ زیادہ متاثرہ جڑیں ہٹائیں اور تیلے کو '
        'ہدایات کے مطابق قابو کریں۔',
    'ہمیشہ تصدیق شدہ بیماری سے پاک (ٹشو کلچر) بیج گنا لگائیں اور موسم '
        'کے شروع میں ہی تیلے کو قابو کریں۔',
  ),
  'Rice___healthy': Advice(
    'No treatment needed. Your rice looks healthy.',
    'Keep the recommended water level and check the leaves every week.',
    'کسی علاج کی ضرورت نہیں۔ آپ کے چاول صحت مند لگ رہے ہیں۔',
    'تجویز کردہ پانی کی سطح رکھیں اور ہر ہفتے پتے چیک کریں۔',
  ),
  'Rice___Bacterial_leaf_blight': Advice(
    'Drain the field for a few days and stop nitrogen for now. Apply a '
        'copper-based bactericide (or streptocycline) as per label '
        'instructions.',
    'Use resistant varieties and clean seed, give balanced fertilizer '
        '(too much nitrogen makes it worse), and keep the field and '
        'bunds clean.',
    'کھیت سے چند دن پانی نکال دیں اور فی الحال نائٹروجن بند کریں۔ '
        'تانبے والی بیکٹیریا مار دوا (یا اسٹریپٹوسائیکلین) ہدایات کے '
        'مطابق استعمال کریں۔',
    'مزاحم اقسام اور صاف بیج استعمال کریں، متوازن کھاد دیں (زیادہ '
        'نائٹروجن بیماری بڑھا دیتی ہے) اور کھیت اور مینڈیں صاف رکھیں۔',
  ),
  'Rice___Brown_spot': Advice(
    'Apply mancozeb fungicide as per label instructions. This disease '
        'often means the soil is hungry - plan a balanced fertilizer '
        'dose (especially potash) for the crop.',
    'Use healthy treated seed, give balanced fertilizer including '
        'potash and silicon, and avoid water stress.',
    'مینکوزیب فنگس مار دوا ہدایات کے مطابق استعمال کریں۔ یہ بیماری '
        'اکثر مٹی کی کمزوری کی نشانی ہے — فصل کے لیے متوازن کھاد (خاص '
        'طور پر پوٹاش) کا بندوبست کریں۔',
    'صحت مند علاج شدہ بیج استعمال کریں، پوٹاش اور سلیکون سمیت متوازن '
        'کھاد دیں اور پانی کی کمی نہ ہونے دیں۔',
  ),
  'Rice___Leaf_blast': Advice(
    'Apply tricyclazole or isoprothiolane fungicide as per label '
        'instructions, as soon as the spindle-shaped spots appear. '
        'Reduce nitrogen for now.',
    'Grow resistant varieties, treat seed before sowing, avoid too '
        'much nitrogen, and keep a steady water level in the field.',
    'جیسے ہی تکلی نما دھبے نظر آئیں، ٹرائی سائیکلازول یا '
        'آئسوپروتھیولین فنگس مار دوا ہدایات کے مطابق استعمال کریں۔ فی '
        'الحال نائٹروجن کم کر دیں۔',
    'مزاحم اقسام اگائیں، بوائی سے پہلے بیج کا علاج کریں، نائٹروجن '
        'زیادہ نہ دیں اور کھیت میں پانی کی سطح یکساں رکھیں۔',
  ),
  'Rice___Leaf_scald': Advice(
    'Apply mancozeb or a copper fungicide as per label instructions '
        'and avoid excess nitrogen.',
    'Use clean seed, keep proper spacing, and give balanced '
        'fertilizer.',
    'مینکوزیب یا تانبے والی فنگس مار دوا ہدایات کے مطابق استعمال '
        'کریں اور نائٹروجن زیادہ نہ دیں۔',
    'صاف بیج استعمال کریں، مناسب فاصلہ رکھیں اور متوازن کھاد دیں۔',
  ),
  'Rice___Sheath_blight': Advice(
    'Apply validamycin or hexaconazole fungicide as per label '
        'instructions, spraying low near the water line where the '
        'disease starts.',
    'Keep wider spacing, avoid too much nitrogen, remove floating '
        'infected debris, and drain the field briefly if the disease '
        'is spreading.',
    'ویلیدامائیسن یا ہیکساکونازول فنگس مار دوا ہدایات کے مطابق '
        'استعمال کریں — سپرے نیچے پانی کی لائن کے قریب کریں جہاں سے '
        'بیماری شروع ہوتی ہے۔',
    'فاصلہ زیادہ رکھیں، نائٹروجن زیادہ نہ دیں، تیرتی ہوئی متاثرہ '
        'باقیات ہٹائیں اور بیماری پھیل رہی ہو تو کھیت سے تھوڑی دیر پانی '
        'نکال دیں۔',
  ),
  'Cotton___healthy': Advice(
    'No treatment needed. Your cotton looks healthy.',
    'Keep normal watering and check the leaves every week, especially '
        'the underside for whiteflies.',
    'کسی علاج کی ضرورت نہیں۔ آپ کی کپاس صحت مند لگ رہی ہے۔',
    'معمول کے مطابق پانی دیں اور ہر ہفتے پتے چیک کریں، خاص طور پر '
        'نیچے کی طرف سفید مکھی کے لیے۔',
  ),
  'Cotton___Bacterial_blight': Advice(
    'Apply a copper-based bactericide as per label instructions. '
        'Remove badly infected leaves and do not work in a wet field - '
        'the bacteria spread with water and hands.',
    'Use acid-delinted, fungicide-treated seed of resistant '
        'varieties, rotate crops, and destroy crop residue after '
        'picking.',
    'تانبے والی بیکٹیریا مار دوا ہدایات کے مطابق استعمال کریں۔ زیادہ '
        'متاثرہ پتے ہٹائیں اور گیلے کھیت میں کام نہ کریں — بیکٹیریا '
        'پانی اور ہاتھوں سے پھیلتے ہیں۔',
    'مزاحم اقسام کا تیزاب سے صاف شدہ، دوا لگا ہوا بیج استعمال کریں، '
        'فصلیں تبدیل کریں اور چنائی کے بعد فصل کی باقیات تلف کریں۔',
  ),
  'Cotton___Leaf_curl_virus': Advice(
    'There is no cure for this virus. Control its carrier, the '
        'whitefly: use yellow sticky traps and, if the attack is '
        'heavy, an insecticide as per label instructions. Remove '
        'badly infected plants early.',
    'Grow resistant varieties (very important for this virus in '
        'Pakistan), sow early, keep the field free of weeds that host '
        'whiteflies, and never keep a ratoon (old) cotton crop.',
    'اس وائرس کا کوئی علاج نہیں۔ اس کو پھیلانے والی سفید مکھی کو '
        'قابو کریں: پیلی چپکنے والی پٹیاں لگائیں اور حملہ زیادہ ہو تو '
        'ہدایات کے مطابق کیڑے مار دوا استعمال کریں۔ زیادہ متاثرہ پودے '
        'جلد ہٹا دیں۔',
    'مزاحم اقسام اگائیں (پاکستان میں اس وائرس کے لیے یہ سب سے اہم '
        'ہے)، جلدی بوائی کریں، کھیت کو سفید مکھی والی جڑی بوٹیوں سے '
        'صاف رکھیں اور کبھی پرانی (رٹون) کپاس نہ رکھیں۔',
  ),
  'Cotton___Fusarium_wilt': Advice(
    'There is no effective spray once the plant wilts. Remove and '
        'destroy wilted plants, and avoid moving soil from this field '
        'to other fields on tools or shoes.',
    'Grow resistant varieties, rotate with non-host crops for a few '
        'years, use treated seed, and keep potash balanced - it helps '
        'the plant resist this fungus.',
    'پودا مرجھا جائے تو کوئی سپرے مؤثر نہیں۔ مرجھائے ہوئے پودے ہٹا '
        'کر تلف کریں اور اس کھیت کی مٹی اوزاروں یا جوتوں کے ساتھ دوسرے '
        'کھیتوں میں نہ لے جائیں۔',
    'مزاحم اقسام اگائیں، چند سال دوسری فصلیں اگائیں، علاج شدہ بیج '
        'استعمال کریں اور پوٹاش متوازن رکھیں — یہ پودے کو اس فنگس سے '
        'لڑنے میں مدد دیتی ہے۔',
  ),
};

// Urdu names for all 32 classes - so a farmer who does not read English
// can still read the result in Urdu mode. Key = exact label from labels.txt.
const Map<String, String> kUrduLabels = {
  'Corn_(maize)___Common_rust_': 'مکئی — عام زنگ (رسٹ)',
  'Corn_(maize)___Northern_Leaf_Blight': 'مکئی — شمالی پتوں کا جھلساؤ',
  'Corn_(maize)___healthy': 'مکئی — صحت مند',
  'Potato___Early_blight': 'آلو — ابتدائی جھلساؤ',
  'Potato___Late_blight': 'آلو — دیر والا جھلساؤ',
  'Potato___healthy': 'آلو — صحت مند',
  'Tomato___Bacterial_spot': 'ٹماٹر — بیکٹیریا والے دھبے',
  'Tomato___Early_blight': 'ٹماٹر — ابتدائی جھلساؤ',
  'Tomato___Late_blight': 'ٹماٹر — دیر والا جھلساؤ',
  'Tomato___Leaf_Mold': 'ٹماٹر — پتوں کی پھپھوندی',
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'ٹماٹر — پیلے مڑے پتوں والا وائرس',
  'Tomato___healthy': 'ٹماٹر — صحت مند',
  'Wheat___healthy': 'گندم — صحت مند',
  'Wheat___Yellow_rust': 'گندم — پیلا زنگ (یلو رسٹ)',
  'Wheat___Brown_rust': 'گندم — بھورا زنگ (براؤن رسٹ)',
  'Wheat___Powdery_mildew': 'گندم — سفوفی پھپھوندی (پاؤڈری ملڈیو)',
  'Wheat___Septoria_leaf_blotch': 'گندم — سیپٹوریا پتوں کے دھبے',
  'Sugarcane___healthy': 'گنا — صحت مند',
  'Sugarcane___Red_rot': 'گنا — سرخ سڑن (ریڈ راٹ)',
  'Sugarcane___Rust': 'گنا — زنگ (رسٹ)',
  'Sugarcane___Mosaic': 'گنا — موزیک وائرس',
  'Sugarcane___Yellow_leaf': 'گنا — پیلے پتے',
  'Rice___healthy': 'چاول — صحت مند',
  'Rice___Bacterial_leaf_blight': 'چاول — بیکٹیریا والا پتوں کا جھلساؤ',
  'Rice___Brown_spot': 'چاول — بھورے دھبے',
  'Rice___Leaf_blast': 'چاول — پتوں کا بلاسٹ',
  'Rice___Leaf_scald': 'چاول — پتوں کا جلنا (لیف اسکالڈ)',
  'Rice___Sheath_blight': 'چاول — شیتھ بلائٹ',
  'Cotton___healthy': 'کپاس — صحت مند',
  'Cotton___Bacterial_blight': 'کپاس — بیکٹیریا والا جھلساؤ',
  'Cotton___Leaf_curl_virus': 'کپاس — پتوں کے مڑنے والا وائرس (لیف کرل)',
  'Cotton___Fusarium_wilt': 'کپاس — فیوزیریم مرجھاؤ',
};

String prettyLabel(String label) =>
    label.replaceAll('___', ' - ').replaceAll('_', ' ').trim();

String displayLabel(String label, bool urdu) =>
    urdu ? (kUrduLabels[label] ?? prettyLabel(label)) : prettyLabel(label);

Advice adviceFor(String label) => kAdvice[label] ??
    const Advice(
      'No advice added for this disease yet. Please consult your local '
          'agriculture extension officer.',
      'Take a clear photo and monitor the plant daily.',
      'اس بیماری کے لیے ابھی مشورہ شامل نہیں کیا گیا۔ براہ کرم مقامی زرعی '
          'افسر سے رابطہ کریں۔',
      'واضح تصویر لیں اور پودے کو روزانہ دیکھیں۔',
    );

// ---------------------------------------------------------------------------
// LOCAL DATABASE - diagnosis history + offline sync queue (Sections 6.2/6.3)
// ---------------------------------------------------------------------------
class HistoryDb {
  static Database? _db;

  static Future<Database> get db async {
    if (_db != null) return _db!;
    final dbPath = await getDatabasesPath();
    _db = await openDatabase(
      p.join(dbPath, 'crop_history.db'),
      version: 2,
      onCreate: (db, _) async {
        await db.execute('''
        CREATE TABLE diagnoses(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          label TEXT, confidence REAL, image_path TEXT,
          diagnosed_at TEXT, synced INTEGER DEFAULT 0
        )''');
        await db.execute(
            'CREATE TABLE session(k TEXT PRIMARY KEY, v TEXT)');
      },
      onUpgrade: (db, oldVersion, _) async {
        if (oldVersion < 2) {
          await db.execute(
              'CREATE TABLE IF NOT EXISTS session(k TEXT PRIMARY KEY, v TEXT)');
        }
      },
    );
    return _db!;
  }

  static Future<int> insert(Map<String, Object?> row) async =>
      (await db).insert('diagnoses', row);

  static Future<List<Map<String, Object?>>> all() async =>
      (await db).query('diagnoses', orderBy: 'id DESC');

  static Future<List<Map<String, Object?>>> unsynced() async =>
      (await db).query('diagnoses', where: 'synced = 0');

  static Future<void> markSynced(int id) async => (await db)
      .update('diagnoses', {'synced': 1}, where: 'id = ?', whereArgs: [id]);

  // ---- Login session (stored on the phone only - app works offline) ----
  static Future<void> saveSession(String name, String phone) async {
    final d = await db;
    await d.insert('session', {'k': 'name', 'v': name},
        conflictAlgorithm: ConflictAlgorithm.replace);
    await d.insert('session', {'k': 'phone', 'v': phone},
        conflictAlgorithm: ConflictAlgorithm.replace);
  }

  static Future<Map<String, String>> session() async {
    final rows = await (await db).query('session');
    return {for (final r in rows) r['k'] as String: (r['v'] ?? '') as String};
  }

  static Future<void> clearSession() async => (await db).delete('session');
}

// ---------------------------------------------------------------------------
// TFLITE CLASSIFIER - on-device inference (Section 6.1/6.2)
//
// IMPORTANT - read this before changing anything:
// The notebook model was trained on RAW pixels (0-255). The MobileNetV3
// scaling to [-1, 1] happens INSIDE the model file. So the app sends the
// photo's raw RGB bytes and must NOT scale them.
//   * int8 model.tflite  -> input tensor is UINT8, output is UINT8
//     (both are converted back with the tensor's scale / zero-point).
//   * float16 model      -> input/output are FLOAT32 raw 0-255 values.
// The code below checks the tensor types and handles both.
// ---------------------------------------------------------------------------
class Classifier {
  Interpreter? _interpreter;
  late List<String> labels;

  Future<void> load() async {
    _interpreter = await Interpreter.fromAsset(kModelAsset);
    final raw = await rootBundle.loadString(kLabelsAsset);
    labels = raw.split('\n').where((e) => e.trim().isNotEmpty).toList();
  }

  /// Returns (label, confidence 0..1) for the best class.
  Future<(String, double)> predict(File imageFile) async {
    final interpreter = _interpreter!;
    final bytes = await imageFile.readAsBytes();
    img.Image? rawDecoded = img.decodeImage(bytes);
    if (rawDecoded == null) throw Exception('Could not read the image.');
    // Camera photos are often SAVED sideways with a rotation flag in
    // the file (the phone screen shows them upright, but the file is
    // rotated). Bake that rotation into the pixels, otherwise the
    // model sees a sideways leaf and answers wrong. Gallery photos
    // have no rotation flag, so this changes nothing for them.
    final decoded = img.bakeOrientation(rawDecoded);
    // Training also squashed photos straight to 224x224, so do the same.
    final resized = img.copyResize(decoded, width: kInputSize, height: kInputSize);

    final inputTensor = interpreter.getInputTensor(0);
    final outputTensor = interpreter.getOutputTensor(0);

    if (inputTensor.type == TensorType.uint8) {
      // ---- INT8 model: fill the input with raw 0-255 pixels, quantized
      // with the input tensor's own scale / zero-point.
      final inParams = inputTensor.params;
      final input = Uint8List(kInputSize * kInputSize * 3);
      int i = 0;
      for (int y = 0; y < kInputSize; y++) {
        for (int x = 0; x < kInputSize; x++) {
          final px = resized.getPixel(x, y);
          for (final v in [px.r, px.g, px.b]) {
            int q = (v / inParams.scale + inParams.zeroPoint).round();
            input[i++] = q.clamp(0, 255);
          }
        }
      }
      inputTensor.data = input;
      interpreter.invoke();
      final outRaw = outputTensor.data; // Uint8List, one byte per class
      final outParams = outputTensor.params;
      int best = 0;
      double bestProb = -1;
      for (int c = 0; c < labels.length; c++) {
        final prob = (outRaw[c] - outParams.zeroPoint) * outParams.scale;
        if (prob > bestProb) {
          bestProb = prob;
          best = c;
        }
      }
      return (labels[best], bestProb);
    } else {
      // ---- FLOAT model (float16 file): raw 0-255 float pixels in,
      // probabilities out. (No [-1, 1] scaling - it is inside the model.)
      final input = Float32List(kInputSize * kInputSize * 3);
      int i = 0;
      for (int y = 0; y < kInputSize; y++) {
        for (int x = 0; x < kInputSize; x++) {
          final px = resized.getPixel(x, y);
          input[i++] = px.r.toDouble();
          input[i++] = px.g.toDouble();
          input[i++] = px.b.toDouble();
        }
      }
      inputTensor.data = input.buffer.asUint8List();
      interpreter.invoke();
      final outBytes = outputTensor.data;
      final outFloats = outBytes.buffer
          .asFloat32List(outBytes.offsetInBytes, labels.length);
      int best = 0;
      for (int c = 1; c < labels.length; c++) {
        if (outFloats[c] > outFloats[best]) best = c;
      }
      return (labels[best], outFloats[best]);
    }
  }
}

// ---------------------------------------------------------------------------
// SYNC - upload queued records when internet is available (Section 6.3)
// ---------------------------------------------------------------------------
Future<int> syncPendingRecords() async {
  int syncedCount = 0;
  final pending = await HistoryDb.unsynced();
  for (final row in pending) {
    try {
      final label = row['label'] as String;
      final crop = label.contains('___') ? label.split('___').first : label;
      final resp = await http
          .post(
            Uri.parse('$kBackendBaseUrl/records'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'device_id': 'flutter-app',
              'crop': crop,
              'disease': label,
              'confidence': row['confidence'],
              'diagnosed_at': row['diagnosed_at'],
              'model_version': kModelVersion,
            }),
          )
          .timeout(const Duration(seconds: 10));
      if (resp.statusCode == 201) {
        await HistoryDb.markSynced(row['id'] as int);
        syncedCount++;
      }
    } catch (_) {
      // Still offline - leave the record in the queue and try next time.
      break;
    }
  }
  return syncedCount;
}

// Online or offline? The app works BOTH ways: diagnosis (model + assistant)
// always runs on the phone offline; when internet is available, pending
// records sync automatically and the home screen shows an Online badge.
Future<bool> isOnline() async {
  try {
    final result = await InternetAddress.lookup('example.com')
        .timeout(const Duration(seconds: 3));
    return result.isNotEmpty && result[0].rawAddress.isNotEmpty;
  } catch (_) {
    return false;
  }
}

// ---------------------------------------------------------------------------
// APP UI
// ---------------------------------------------------------------------------
void main() => runApp(const CropApp());

class CropApp extends StatelessWidget {
  const CropApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'CropCare AI',
        theme: ThemeData(colorSchemeSeed: Colors.green, useMaterial3: true),
        home: const AuthGate(),
      );
}

class HomeScreen extends StatefulWidget {
  final String userName;
  final bool urduStart;
  const HomeScreen({super.key, this.userName = '', this.urduStart = false});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final Classifier _classifier = Classifier();
  final ImagePicker _picker = ImagePicker();
  bool _modelReady = false;
  bool _urdu = false;
  File? _image;
  String? _resultText;
  Advice? _advice;
  String? _lastLabel;
  String _phone = '';
  int _diagCount = 0;
  bool? _online;
  bool _scanning = false;
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _urdu = widget.urduStart;
    _loadProfile();
    _checkOnline();
    _classifier.load().then((_) {
      setState(() => _modelReady = true);
      syncPendingRecords(); // try to upload old records at startup
    });
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _loadProfile() async {
    final s = await HistoryDb.session();
    final rows = await HistoryDb.all();
    if (!mounted) return;
    setState(() {
      _phone = s['phone'] ?? '';
      _diagCount = rows.length;
    });
  }

  Future<void> _checkOnline() async {
    final on = await isOnline();
    if (!mounted) return;
    setState(() => _online = on);
  }

  Future<void> _pickAndDiagnose(ImageSource source) async {
    final picked = await _picker.pickImage(source: source, imageQuality: 90);
    if (picked == null) return;
    await _diagnose(File(picked.path));
  }

  /// Camera button: open the SCANNER-style camera (live preview with a
  /// leaf frame and a moving scan line). If the scanner camera cannot
  /// start on this phone, the scanner screen falls back by itself to
  /// the normal camera, so this always ends with a photo or nothing.
  Future<void> _openScanner() async {
    final file = await Navigator.push<File>(
      context,
      MaterialPageRoute(builder: (_) => ScannerScreen(urdu: _urdu)),
    );
    if (file == null) return;
    await _diagnose(file);
  }

  Future<void> _diagnose(File file) async {
    final sw = Stopwatch()..start();
    setState(() {
      _image = file;
      _scanning = true;
      _resultText = _urdu ? 'پتا اسکین ہو رہا ہے…' : 'Scanning leaf…';
      _advice = null;
    });

    final (label, confidence) = await _classifier.predict(file);
    _lastLabel = label;
    await HistoryDb.insert({
      'label': label,
      'confidence': confidence,
      'image_path': file.path,
      'diagnosed_at': DateTime.now().toIso8601String(),
      'synced': 0,
    });
    syncPendingRecords();
    _checkOnline();
    _diagCount++;

    // Keep the scanning animation on screen briefly, so the user can
    // see the leaf is really being scanned.
    final left = 1400 - sw.elapsedMilliseconds;
    if (left > 0) await Future.delayed(Duration(milliseconds: left));
    if (!mounted) return;

    setState(() {
      _scanning = false;
      if (confidence < kConfidenceThreshold) {
        // Proposal 8.3: do not give a misleading answer.
        _resultText = _urdu
            ? 'یقین نہیں ہے (${(confidence * 100).toStringAsFixed(0)}٪)۔ براہ کرم پتے کی واضح اور قریب سے تصویر لیں۔'
            : 'Uncertain (${(confidence * 100).toStringAsFixed(0)}%). '
                'Please take a clearer, closer photo of a single leaf.';
        _advice = null;
      } else {
        _resultText = _urdu
            ? 'نتیجہ: ${displayLabel(label, true)}\nاعتماد: ${(confidence * 100).toStringAsFixed(1)}٪'
            : 'Result: ${displayLabel(label, false)}\nConfidence: ${(confidence * 100).toStringAsFixed(1)}%';
        _advice = adviceFor(label);
      }
    });
  }

  /// "Back - scan another leaf": clear the photo and result and go
  /// back to the top of the home screen.
  void _resetScan() {
    setState(() {
      _image = null;
      _resultText = null;
      _advice = null;
    });
    _scrollController.animateTo(0,
        duration: const Duration(milliseconds: 400), curve: Curves.easeOut);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
        title: Text(_urdu ? 'فصل کی بیماری کی پہچان' : 'CropCare AI'),
        actions: [
          TextButton(
            onPressed: () => setState(() => _urdu = !_urdu),
            child: Text(_urdu ? 'EN' : 'اردو',
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
          ),
          IconButton(
            icon: const Icon(Icons.account_circle),
            tooltip: _urdu ? 'میری پروفائل' : 'My Profile',
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(
                  builder: (_) =>
                      ProfileScreen(urdu: _urdu, userName: widget.userName)),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.history),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => HistoryScreen(urdu: _urdu)),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.logout),
            tooltip: _urdu ? 'لاگ آؤٹ' : 'Logout',
            onPressed: () async {
              await HistoryDb.clearSession();
              if (!context.mounted) return;
              Navigator.pushAndRemoveUntil(
                context,
                MaterialPageRoute(builder: (_) => const LoginScreen()),
                (_) => false,
              );
            },
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        icon: const Icon(Icons.smart_toy),
        label: Text(_urdu ? 'AI مددگار' : 'AI Assistant'),
        onPressed: () => Navigator.push(
          context,
          MaterialPageRoute(
              builder: (_) =>
                  AssistantScreen(urdu: _urdu, lastLabel: _lastLabel)),
        ),
      ),
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/plants_bg.jpg'),
            fit: BoxFit.cover,
          ),
        ),
        child: !_modelReady
            ? const Center(child: CircularProgressIndicator())
            : ListView(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              children: [
                Card(
                  color: Colors.white.withOpacity(0.92),
                  child: InkWell(
                    onTap: () => Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (_) => ProfileScreen(
                              urdu: _urdu, userName: widget.userName)),
                    ).then((_) => _loadProfile()),
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: Row(
                        children: [
                          CircleAvatar(
                            radius: 26,
                            backgroundColor: Colors.green.shade700,
                            child: Text(
                              widget.userName.isEmpty
                                  ? 'G'
                                  : widget.userName[0].toUpperCase(),
                              style: const TextStyle(
                                  fontSize: 22, color: Colors.white),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  widget.userName.isEmpty
                                      ? (_urdu ? 'مہمان صارف' : 'Guest User')
                                      : widget.userName,
                                  style: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold),
                                ),
                                Text(
                                  _phone.isEmpty
                                      ? (_urdu
                                          ? 'پروفائل دیکھنے کے لیے دبائیں'
                                          : 'Tap to view profile')
                                      : _phone,
                                  style: const TextStyle(
                                      fontSize: 12, color: Colors.grey),
                                ),
                              ],
                            ),
                          ),
                          Chip(
                            label: Text(_urdu
                                ? '$_diagCount تشخیصیں'
                                : '$_diagCount diagnoses'),
                          ),
                          const Icon(Icons.chevron_right),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                InkWell(
                  onTap: _checkOnline,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: _online == true
                          ? Colors.green.shade600
                          : Colors.grey.shade600,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                            _online == true
                                ? Icons.cloud_done
                                : Icons.cloud_off,
                            color: Colors.white,
                            size: 18),
                        const SizedBox(width: 8),
                        Text(
                          _online == null
                              ? (_urdu ? 'چیک ہو رہا ہے…' : 'Checking…')
                              : _online == true
                                  ? (_urdu
                                      ? 'آن لائن — تشخیص + ریکارڈ sync دونوں چلیں گے'
                                      : 'Online - diagnosis + record sync both work')
                                  : (_urdu
                                      ? 'آف لائن — تشخیص چلے گی، ریکارڈ بعد میں sync ہوں گے'
                                      : 'Offline - diagnosis works, records sync later'),
                          style: const TextStyle(
                              color: Colors.white, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                Card(
                  color: Colors.white.withOpacity(0.85),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Row(
                      children: [
                        Image.asset('assets/logo.png', width: 52, height: 52),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            _urdu
                                ? (widget.userName.isEmpty
                                    ? 'پتے کی تصویر لیں، بیماری جانیں'
                                    : 'خوش آمدید، ${widget.userName} — پتے کی تصویر لیں، بیماری جانیں')
                                : (widget.userName.isEmpty
                                    ? 'Take a leaf photo, know the disease'
                                    : 'Welcome, ${widget.userName} — take a leaf photo, know the disease'),
                            style: const TextStyle(
                                fontSize: 16, fontWeight: FontWeight.bold, color: Colors.green),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                if (_image != null)
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Stack(
                      children: [
                        Image.file(_image!,
                            height: 260,
                            width: double.infinity,
                            fit: BoxFit.cover),
                        if (_scanning)
                          const Positioned.fill(child: ScanLineOverlay()),
                      ],
                    ),
                  ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: FilledButton.icon(
                        icon: const Icon(Icons.qr_code_scanner),
                        label: Text(_urdu ? 'اسکینر' : 'Scanner'),
                        onPressed: _openScanner,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: OutlinedButton.icon(
                        icon: const Icon(Icons.photo_library),
                        label: Text(_urdu ? 'گیلری' : 'Gallery'),
                        onPressed: () => _pickAndDiagnose(ImageSource.gallery),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                if (_resultText != null)
                  Text(_resultText!,
                      style: Theme.of(context).textTheme.titleMedium),
                if (_advice != null) ...[
                  const SizedBox(height: 12),
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(_urdu ? 'علاج:' : 'Treatment:',
                              style: const TextStyle(fontWeight: FontWeight.bold)),
                          Text(_urdu ? _advice!.treatmentUr : _advice!.treatmentEn),
                          const SizedBox(height: 8),
                          Text(_urdu ? 'احتیاط:' : 'Prevention:',
                              style: const TextStyle(fontWeight: FontWeight.bold)),
                          Text(_urdu ? _advice!.preventionUr : _advice!.preventionEn),
                        ],
                      ),
                    ),
                  ),
                ],
                if (_image != null && !_scanning) ...[
                  const SizedBox(height: 16),
                  OutlinedButton.icon(
                    icon: const Icon(Icons.arrow_back),
                    label: Text(_urdu
                        ? 'واپس — ایک اور پتا اسکین کریں'
                        : 'Back - scan another leaf'),
                    onPressed: _resetScan,
                  ),
                ],
                const SizedBox(height: 12),
                Text(
                  _urdu
                      ? 'یہ ایپ انٹرنیٹ کے بغیر بھی کام کرتی ہے۔'
                      : 'This app works fully offline. Records sync when internet returns.',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
      ),
    );
  }
}


// ---------------------------------------------------------------------------
// LOGIN - a DIFFERENT interface from the home screen on purpose:
// dark green gradient + big logo in a white circle + a white login card.
// Login is stored ONLY on this phone (SQLite session table), so the app
// still works fully offline. "Continue as Guest" is kept for farmers who
// have no account - they must never be blocked from a diagnosis.
// ---------------------------------------------------------------------------
class AuthGate extends StatefulWidget {
  const AuthGate({super.key});
  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  @override
  void initState() {
    super.initState();
    HistoryDb.session().then((s) {
      if (!mounted) return;
      final name = s['name'] ?? '';
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => name.isEmpty
              ? const LoginScreen()
              : HomeScreen(userName: name),
        ),
      );
    });
  }

  @override
  Widget build(BuildContext context) => const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
}

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});
  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _name = TextEditingController();
  final _phone = TextEditingController();
  bool _urdu = false;
  String? _error;

  void _goHome(String name) {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
          builder: (_) => HomeScreen(userName: name, urduStart: _urdu)),
    );
  }

  Future<void> _login() async {
    final name = _name.text.trim();
    final phone = _phone.text.replaceAll(RegExp(r'[^0-9]'), '');
    if (name.isEmpty || phone.length < 10) {
      setState(() => _error = _urdu
          ? 'براہ کرم اپنا نام اور درست موبائل نمبر (کم از کم 10 ہندسے) لکھیں۔'
          : 'Please enter your name and a correct mobile number (at least 10 digits).');
      return;
    }
    await HistoryDb.saveSession(name, phone);
    if (!mounted) return;
    _goHome(name);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.green.shade900, Colors.green.shade600],
          ),
        ),
        child: SafeArea(
          child: ListView(
            padding: const EdgeInsets.all(24),
            children: [
              Align(
                alignment: Alignment.centerRight,
                child: TextButton(
                  onPressed: () => setState(() => _urdu = !_urdu),
                  child: Text(_urdu ? 'EN' : 'اردو',
                      style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 16)),
                ),
              ),
              const SizedBox(height: 8),
              Center(
                child: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: const BoxDecoration(
                      color: Colors.white, shape: BoxShape.circle),
                  child: ClipOval(
                    child: Image.asset('assets/logo.png',
                        width: 110, height: 110, fit: BoxFit.cover),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              const Center(
                child: Text('CropCare AI',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 30,
                        fontWeight: FontWeight.bold)),
              ),
              Center(
                child: Text(
                  _urdu
                      ? 'پتے کی تصویر — بیماری کی پہچان'
                      : 'Leaf photo in, disease name out',
                  style: const TextStyle(color: Colors.white70, fontSize: 15),
                ),
              ),
              const SizedBox(height: 28),
              Card(
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20)),
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    children: [
                      Text(_urdu ? 'لاگ ان' : 'Login',
                          style: const TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.green)),
                      const SizedBox(height: 16),
                      TextField(
                        controller: _name,
                        decoration: InputDecoration(
                          prefixIcon: const Icon(Icons.person),
                          labelText: _urdu ? 'آپ کا نام' : 'Your name',
                          border: const OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _phone,
                        keyboardType: TextInputType.phone,
                        decoration: InputDecoration(
                          prefixIcon: const Icon(Icons.phone),
                          labelText:
                              _urdu ? 'موبائل نمبر' : 'Mobile number',
                          border: const OutlineInputBorder(),
                        ),
                      ),
                      if (_error != null) ...[
                        const SizedBox(height: 10),
                        Text(_error!,
                            style: const TextStyle(color: Colors.red)),
                      ],
                      const SizedBox(height: 18),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton(
                          onPressed: _login,
                          child: Text(_urdu ? 'لاگ ان کریں' : 'Login'),
                        ),
                      ),
                      const SizedBox(height: 8),
                      TextButton(
                        onPressed: () => _goHome(''),
                        child: Text(_urdu
                            ? 'بغیر لاگ ان کے استعمال کریں (مہمان)'
                            : 'Continue as Guest (no login)'),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Center(
                child: Text(
                  _urdu
                      ? 'لاگ ان صرف اسی فون میں محفوظ ہوتا ہے — انٹرنیٹ کی ضرورت نہیں۔'
                      : 'Login is saved on this phone only - no internet needed.',
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}


// ---------------------------------------------------------------------------
// AI ASSISTANT - offline chat help, built into the app (no internet, no
// API key). It answers from the same advice knowledge as the diagnosis
// screen: all 32 diseases (treatment + prevention), photo tips, low
// confidence, watering, fertilizer and general prevention - in English
// or Urdu. Viva point: the assistant works 100% offline, like the model.
// ---------------------------------------------------------------------------
class AssistantScreen extends StatefulWidget {
  final bool urdu;
  final String? lastLabel;
  const AssistantScreen({super.key, this.urdu = false, this.lastLabel});
  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen> {
  final _input = TextEditingController();
  final List<(String, bool)> _msgs = []; // (text, isUser)
  final SpeechToText _speech = SpeechToText();
  final FlutterTts _tts = FlutterTts();
  bool _listening = false;
  bool _speakOn = true;

  @override
  void dispose() {
    _tts.stop();
    _speech.stop();
    super.dispose();
  }

  Future<void> _speak(String text) async {
    if (!_speakOn) return;
    try {
      await _tts.setLanguage(_u ? 'ur-PK' : 'en-US');
      await _tts.setSpeechRate(0.48);
      await _tts.speak(text);
    } catch (_) {
      // Phone has no voice for this language - text answer is still shown.
    }
  }

  Future<void> _listen() async {
    if (_listening) {
      await _speech.stop();
      setState(() => _listening = false);
      return;
    }
    final available = await _speech.initialize(
      onStatus: (s) {
        if (s == 'done' || s == 'notListening') {
          setState(() => _listening = false);
        }
      },
      onError: (_) => setState(() => _listening = false),
    );
    if (!available) {
      setState(() => _msgs.add((
            _u
                ? 'اس فون پر آواز کی سہولت دستیاب نہیں — براہ کرم سوال لکھ کر بھیجیں۔'
                : 'Voice is not available on this phone - please type your question instead.',
            false
          )));
      return;
    }
    setState(() => _listening = true);
    await _speech.listen(
      localeId: _u ? 'ur_PK' : 'en_US',
      onResult: (r) {
        setState(() => _input.text = r.recognizedWords);
        if (r.finalResult && r.recognizedWords.trim().isNotEmpty) {
          setState(() => _listening = false);
          _send(r.recognizedWords);
        }
      },
    );
  }

  bool get _u => widget.urdu;

  @override
  void initState() {
    super.initState();
    _msgs.add((
      _u
          ? 'السلام علیکم! میں آپ کا AI مددگار ہوں۔ آپ مجھ سے کسی بھی بیماری کے علاج، احتیاط، تصویر لینے کے طریقے یا کھاد/پانی کے بارے میں پوچھ سکتے ہیں۔ نیچے دیے گئے سوالوں پر بھی دبا سکتے ہیں۔ اگر لکھنا مشکل ہو تو مائیک دبا کر بولیں — میں جواب بھی بول کر سناؤں گا۔'
          : 'Hello! I am your AI Assistant. Ask me about any disease treatment, prevention, how to take a good photo, fertilizer or watering. You can type, tap a ready question, or press the mic and SPEAK - I will also speak my answer back to you.',
      false
    ));
    final last = widget.lastLabel;
    if (last != null) {
      _msgs.add((
        _u
            ? 'آپ کی آخری تشخیص: ${displayLabel(last, true)}۔ اس کے علاج کے لیے نیچے والا بٹن دبائیں۔'
            : 'Your last diagnosis was: ${displayLabel(last, false)}. Tap the button below for its treatment.',
        false
      ));
    }
  }

  String _diseaseReply(String label, bool urdu) {
    final a = adviceFor(label);
    return urdu
        ? '${displayLabel(label, true)}\n\nعلاج: ${a.treatmentUr}\n\nاحتیاط: ${a.preventionUr}'
        : '${displayLabel(label, false)}\n\nTreatment: ${a.treatmentEn}\n\nPrevention: ${a.preventionEn}';
  }

  String _answer(String raw) {
    final q = raw.toLowerCase();
    final romanUrdu = ['ilaj', 'batao', 'batain', 'kese', 'kesay', 'kaise', 'kya', 'bimari', 'chahiye', 'mera', 'meri', 'kyun', 'kahan', 'madad', 'sawal', 'pooch', 'acha', 'theek', 'karo', 'hota', 'hai', 'hain', 'muje', 'mujhe', 'aap', 'yeh', 'woh']
        .any((w) => q.split(RegExp(r'[^a-z]+')).contains(w));
    final urdu = _u || romanUrdu || RegExp(r'[\u0600-\u06FF]').hasMatch(raw);
    bool has(List<String> words) => words.any((w) => q.contains(w));

    // --- find a specific disease: crop word + disease word ---
    String? crop;
    if (has(['tomato', 'ٹماٹر', 'tamatar'])) crop = 'Tomato';
    if (has(['potato', 'آلو', 'aalu', 'aloo'])) crop = 'Potato';
    if (has(['corn', 'maize', 'مکئی', 'makai'])) crop = 'Corn_(maize)';
    if (has(['wheat', 'گندم', 'gandum'])) crop = 'Wheat';
    if (has(['sugarcane', 'sugar cane', 'گنا', 'ganna'])) crop = 'Sugarcane';
    if (has(['rice', 'چاول', 'chawal', 'دھان'])) crop = 'Rice';
    if (has(['cotton', 'کپاس', 'kapas'])) crop = 'Cotton';
    String? disease;
    if (has(['healthy', 'صحت مند', 'sehat'])) disease = 'healthy';
    if (has(['early', 'ابتدائی'])) disease = 'Early_blight';
    if (has(['late', 'دیر', 'پچھیتا'])) disease = 'Late_blight';
    if (has(['northern', 'شمالی'])) disease = 'Northern_Leaf_Blight';
    if (has(['septoria', 'سیپٹوریا'])) disease = 'Septoria_leaf_blotch';
    if (has(['powdery', 'mildew', 'سفوفی'])) disease = 'Powdery_mildew';
    if (has(['red rot', 'سرخ سڑن'])) disease = 'Red_rot';
    if (has(['mosaic', 'موزیک'])) disease = 'Mosaic';
    if (has(['sheath', 'شیتھ'])) disease = 'Sheath_blight';
    if (has(['scald', 'اسکالڈ'])) disease = 'Leaf_scald';
    if (has(['blast', 'بلاسٹ'])) disease = 'Leaf_blast';
    if (has(['brown spot', 'بھورے دھبے'])) disease = 'Brown_spot';
    if (has(['fusarium', 'wilt', 'فیوزیریم', 'مرجھاؤ'])) {
      disease = 'Fusarium_wilt';
    }
    if (has(['yellow leaf', 'پیلے پتے']) && crop == 'Sugarcane') {
      disease = 'Yellow_leaf';
    }
    if (has(['rust', 'زنگ'])) {
      if (has(['yellow', 'stripe', 'پیلا'])) {
        disease = 'Yellow_rust';
      } else if (has(['brown', 'بھورا'])) {
        disease = 'Brown_rust';
      } else if (crop == 'Sugarcane') {
        disease = 'Rust';
      } else if (crop == 'Wheat') {
        disease = 'Yellow_rust';
      } else {
        disease = 'Common_rust_';
      }
    }
    if (has(['bacterial', 'bacteria', 'بیکٹیریا'])) {
      if (crop == 'Rice') {
        disease = 'Bacterial_leaf_blight';
      } else if (crop == 'Cotton') {
        disease = 'Bacterial_blight';
      } else {
        disease = 'Bacterial_spot';
      }
    }
    if (has(['spot', 'دھبے']) && disease == null) disease = 'Bacterial_spot';
    if (has(['mold', 'پھپھوندی', 'phaphundi'])) disease = 'Leaf_Mold';
    if (crop == 'Cotton' && has(['curl', 'virus', 'وائرس', 'مڑے'])) {
      disease = 'Leaf_curl_virus';
    } else if (disease == null &&
        has(['yellow', 'curl', 'virus', 'پیلے', 'وائرس', 'مڑے'])) {
      disease = 'Tomato_Yellow_Leaf_Curl_Virus';
      crop = 'Tomato';
    }
    if (crop != null && disease != null) {
      for (final key in kAdvice.keys) {
        if (key.startsWith(crop!) && key.contains(disease!)) {
          return _diseaseReply(key, urdu);
        }
      }
    }
    if (disease != null && crop == null && disease != 'healthy') {
      return urdu
          ? 'یہ بیماری کئی فصلوں میں ہو سکتی ہے۔ براہ کرم فصل کا نام بھی لکھیں — مثلاً "ٹماٹر ابتدائی جھلساؤ" یا "آلو دیر والا جھلساؤ"۔'
          : 'This disease can occur in more than one crop. Please also write the crop name - for example "tomato early blight" or "potato late blight".';
    }

    if (has(['hello', 'hi', 'salam', 'سلام', 'ہیلو'])) {
      return urdu
          ? 'وعلیکم السلام! آپ کس فصل یا بیماری کے بارے میں جاننا چاہتے ہیں؟'
          : 'Hello! Which crop or disease do you want to know about?';
    }
    if (has(['photo', 'tasveer', 'تصویر', 'camera', 'کیمرہ'])) {
      return urdu
          ? 'اچھی تصویر کے لیے: 1) صرف ایک پتا قریب سے لیں 2) دن کی روشنی میں تصویر لیں، سایہ نہ ہو 3) پتا صاف اور دھندلا نہ ہو 4) بیماری والے حصے پر فوکس کریں۔ اچھی تصویر سے نتیجہ زیادہ درست آتا ہے۔'
          : 'For a good photo: 1) Take one leaf, close up 2) Use daylight, avoid shadow 3) Keep the leaf clear, not blurry 4) Focus on the diseased part. A good photo gives a more correct result.';
    }
    if (has(['confidence', 'uncertain', 'یقین', 'غیر یقینی', 'result wrong', 'غلط'])) {
      return urdu
          ? 'اگر اعتماد (confidence) 60٪ سے کم ہو تو ایپ جان بوجھ کر جواب نہیں دیتی — غلط مشورے سے بچنے کے لیے۔ پتے کی زیادہ قریب اور صاف تصویر دن کی روشنی میں دوبارہ لیں۔'
          : 'If confidence is below 60%, the app honestly does not give an answer - to avoid wrong advice. Take a closer, clearer photo of the leaf in daylight and try again.';
    }
    if (has(['offline', 'internet', 'انٹرنیٹ', 'net'])) {
      return urdu
          ? 'یہ ایپ اور یہ AI مددگار دونوں انٹرنیٹ کے بغیر کام کرتے ہیں۔ تشخیص بھی فون کے اندر ہی ہوتی ہے۔'
          : 'This app works BOTH online and offline. Offline: diagnosis, history and this assistant all work inside the phone. Online: the same things work, plus your saved records sync automatically. Look at the Online/Offline badge on the home screen.';
    }
    if (has(['khad', 'fertilizer', 'کھاد'])) {
      return urdu
          ? 'عام اصول: متوازن کھاد (NPK) فصل کی ضرورت کے مطابق دیں، ضرورت سے زیادہ نائٹروجن پتوں کی بیماریاں بڑھا سکتی ہے۔ درست مقدار کے لیے مقامی زرعی افسر یا مٹی کا ٹیسٹ بہتر ہے۔'
          : 'General rule: give balanced fertilizer (NPK) as the crop needs. Too much nitrogen can increase leaf diseases. For the exact dose, a soil test or your local agriculture officer is best.';
    }
    if (has(['pani', 'water', 'پانی', 'آبپاشی', 'irrigation'])) {
      return urdu
          ? 'پانی جڑوں میں دیں، پتوں پر نہ ڈالیں — گیلے پتے بیماری کو دعوت دیتے ہیں۔ صبح کے وقت پانی دینا بہتر ہے اور پودوں میں ہوا کے لیے فاصلہ رکھیں۔'
          : 'Water at the roots, not on the leaves - wet leaves invite disease. Morning watering is better, and keep spacing between plants for air flow.';
    }
    if (has(['prevention', 'bachao', 'احتیاط', 'روک تھام', 'protect'])) {
      return urdu
          ? 'بیماری سے بچاؤ کے عام اصول: 1) بیماری سے پاک بیج 2) ہر موسم میں فصل تبدیل کریں 3) پودوں میں فاصلہ رکھیں 4) پانی جڑوں میں دیں 5) متاثرہ پتے فوراً ہٹائیں۔ کسی خاص بیماری کے لیے اس کا نام لکھیں۔'
          : 'General prevention: 1) Use disease-free seed 2) Rotate crops each season 3) Keep spacing between plants 4) Water at the roots 5) Remove infected leaves quickly. For a specific disease, write its name.';
    }
    if (has(['treatment', 'ilaj', 'علاج', 'spray', 'سپرے', 'dawa', 'دوا', 'bimari', 'بیماری', 'disease'])) {
      return urdu
          ? 'علاج بیماری کے نام پر منحصر ہے۔ آپ فصل اور بیماری کا نام لکھیں — مثلاً "ٹماٹر لیف مولڈ" — یا پہلے Camera سے پتے کی تصویر لے کر تشخیص کر لیں، پھر علاج خود ہی اسی اسکرین پر آ جائے گا۔'
          : 'Treatment depends on the exact disease. Write the crop and disease name - for example "tomato leaf mold" - or first take a leaf photo with Camera; the treatment appears on the result screen itself.';
    }
    if (has(['history', 'تاریخ', 'ریکارڈ', 'record'])) {
      return urdu
          ? 'آپ کی ساری پچھلی تشخیصیں اوپر والے گھڑی (history) والے بٹن میں محفوظ ہوتی ہیں — انٹرنیٹ کے بغیر بھی۔'
          : 'All your past diagnoses are saved under the clock (history) button at the top - even without internet.';
    }
    return urdu
        ? 'معذرت، یہ سوال میری سمجھ سے باہر ہے۔ آپ فصل اور بیماری کا نام لکھ کر پوچھیں — مثلاً "آلو ابتدائی جھلساؤ کا علاج" — یا نیچے دیے گئے تیار سوالوں میں سے ایک دبائیں۔'
        : 'Sorry, I did not understand that question. Ask with a crop and disease name - for example "treatment of potato early blight" - or tap one of the ready questions below.';
  }

  void _send(String text) {
    final t = text.trim();
    if (t.isEmpty) return;
    final reply = _answer(t);
    setState(() {
      _msgs.add((t, true));
      _msgs.add((reply, false));
    });
    _input.clear();
    _speak(reply);
  }

  @override
  Widget build(BuildContext context) {
    final last = widget.lastLabel;
    final chips = <String>[
      if (last != null) (_u ? 'میری آخری بیماری کا علاج' : 'Treatment of my last result'),
      _u ? 'ٹماٹر ابتدائی جھلساؤ کا علاج' : 'Tomato early blight treatment',
      _u ? 'آلو دیر والا جھلساؤ' : 'Potato late blight',
      _u ? 'گندم پیلا زنگ' : 'Wheat yellow rust',
      _u ? 'کپاس لیف کرل وائرس' : 'Cotton leaf curl virus',
      _u ? 'اچھی تصویر کیسے لوں؟' : 'How to take a good photo?',
      _u ? 'بیماری سے بچاؤ کیسے ہو؟' : 'How to prevent disease?',
    ];
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          tooltip: 'Back',
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(_u ? 'AI مددگار' : 'AI Assistant'),
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: Icon(_speakOn ? Icons.volume_up : Icons.volume_off),
            tooltip: _u ? 'آواز آن/آف' : 'Voice on/off',
            onPressed: () {
              setState(() => _speakOn = !_speakOn);
              if (!_speakOn) _tts.stop();
            },
          ),
        ],
      ),
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/plants_bg.jpg'),
            fit: BoxFit.cover,
          ),
        ),
        child: Column(
          children: [
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.all(12),
                itemCount: _msgs.length,
                itemBuilder: (_, i) {
                  final (text, isUser) = _msgs[i];
                  return Align(
                    alignment:
                        isUser ? Alignment.centerRight : Alignment.centerLeft,
                    child: Container(
                      margin: const EdgeInsets.symmetric(vertical: 4),
                      padding: const EdgeInsets.all(12),
                      constraints: BoxConstraints(
                          maxWidth: MediaQuery.of(context).size.width * 0.82),
                      decoration: BoxDecoration(
                        color: isUser
                            ? Colors.green.shade600
                            : Colors.white.withOpacity(0.95),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Text(text,
                          style: TextStyle(
                              color: isUser ? Colors.white : Colors.black87)),
                    ),
                  );
                },
              ),
            ),
            SizedBox(
              height: 44,
              child: ListView(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 8),
                children: [
                  for (final c in chips)
                    Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: ActionChip(
                        label: Text(c),
                        onPressed: () => _send(
                            c ==
                                    (_u
                                        ? 'میری آخری بیماری کا علاج'
                                        : 'Treatment of my last result')
                                ? 'treatment of ${last ?? ''}'
                                : c),
                      ),
                    ),
                ],
              ),
            ),
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(8),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _input,
                        onSubmitted: _send,
                        decoration: InputDecoration(
                          hintText: _u
                              ? 'اپنا سوال یہاں لکھیں…'
                              : 'Type your question here…',
                          filled: true,
                          fillColor: Colors.white,
                          border: const OutlineInputBorder(),
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    IconButton.filledTonal(
                      icon: Icon(_listening ? Icons.mic_off : Icons.mic,
                          color: _listening ? Colors.red : null),
                      tooltip: _u ? 'بول کر پوچھیں' : 'Ask by voice',
                      onPressed: _listen,
                    ),
                    const SizedBox(width: 4),
                    IconButton.filled(
                      icon: const Icon(Icons.send),
                      onPressed: () => _send(_input.text),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}


// ---------------------------------------------------------------------------
// PROFILE - user details + everything about their use of the app:
// name & mobile (from login, stored on the phone), total diagnoses,
// pending-sync count, last diagnosis, and shortcuts to History, the AI
// Assistant and Logout. Opened from the person icon on the home screen.
// ---------------------------------------------------------------------------
class ProfileScreen extends StatelessWidget {
  final bool urdu;
  final String userName;
  const ProfileScreen({super.key, this.urdu = false, this.userName = ''});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          tooltip: 'Back',
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(urdu ? 'میری پروفائل' : 'My Profile'),
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
      ),
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/plants_bg.jpg'),
            fit: BoxFit.cover,
          ),
        ),
        child: FutureBuilder(
          future: Future.wait([HistoryDb.session(), HistoryDb.all()]),
          builder: (context, snap) {
            if (!snap.hasData) {
              return const Center(child: CircularProgressIndicator());
            }
            final session = snap.data![0] as Map<String, String>;
            final rows = snap.data![1] as List<Map<String, Object?>>;
            final name = (session['name'] ?? '').isNotEmpty
                ? session['name']!
                : userName;
            final phone = session['phone'] ?? '';
            final pending = rows.where((r) => (r['synced'] as int) == 0).length;
            final last = rows.isNotEmpty ? rows.first : null;
            return ListView(
              padding: const EdgeInsets.all(16),
              children: [
                Center(
                  child: CircleAvatar(
                    radius: 44,
                    backgroundColor: Colors.green.shade700,
                    child: Text(
                      name.isEmpty ? 'G' : name[0].toUpperCase(),
                      style:
                          const TextStyle(fontSize: 36, color: Colors.white),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                Center(
                  child: Text(
                    name.isEmpty
                        ? (urdu ? 'مہمان صارف' : 'Guest User')
                        : name,
                    style: const TextStyle(
                        fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                ),
                if (phone.isNotEmpty)
                  Center(child: Text(phone)),
                const SizedBox(height: 16),
                Card(
                  color: Colors.white.withOpacity(0.92),
                  child: Column(
                    children: [
                      ListTile(
                        leading: const Icon(Icons.eco, color: Colors.green),
                        title: Text(urdu
                            ? 'کل تشخیصیں: ${rows.length}'
                            : 'Total diagnoses: ${rows.length}'),
                      ),
                      ListTile(
                        leading:
                            const Icon(Icons.cloud_off, color: Colors.grey),
                        title: Text(urdu
                            ? 'انٹرنیٹ کے منتظر ریکارڈ: $pending'
                            : 'Records waiting to sync: $pending'),
                      ),
                      if (last != null)
                        ListTile(
                          leading: const Icon(Icons.history,
                              color: Colors.green),
                          title: Text(urdu
                              ? 'آخری تشخیص: ${displayLabel('${last['label']}', true)}'
                              : 'Last diagnosis: ${displayLabel('${last['label']}', false)}'),
                          subtitle: Text('${last['diagnosed_at']}'),
                        ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                FilledButton.tonalIcon(
                  icon: const Icon(Icons.sync),
                  label: Text(urdu ? 'ابھی Sync کریں' : 'Sync Now'),
                  onPressed: () async {
                    final on = await isOnline();
                    if (!context.mounted) return;
                    if (!on) {
                      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                          content: Text(urdu
                              ? 'ابھی انٹرنیٹ نہیں ہے — ریکارڈ فون میں محفوظ ہیں، انٹرنیٹ آتے ہی sync ہو جائیں گے۔'
                              : 'No internet right now - records are safe on the phone and will sync when internet returns.')));
                      return;
                    }
                    await syncPendingRecords();
                    if (!context.mounted) return;
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(
                          builder: (_) =>
                              ProfileScreen(urdu: urdu, userName: userName)),
                    );
                  },
                ),
                const SizedBox(height: 8),
                FilledButton.icon(
                  icon: const Icon(Icons.history),
                  label: Text(urdu ? 'پوری تاریخ دیکھیں' : 'View Full History'),
                  onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (_) => HistoryScreen(urdu: urdu)),
                  ),
                ),
                const SizedBox(height: 8),
                OutlinedButton.icon(
                  icon: const Icon(Icons.smart_toy),
                  label: Text(urdu ? 'AI مددگار' : 'AI Assistant'),
                  onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (_) => AssistantScreen(urdu: urdu)),
                  ),
                ),
                const SizedBox(height: 8),
                OutlinedButton.icon(
                  icon: const Icon(Icons.arrow_back),
                  label: Text(urdu ? 'واپس' : 'Back'),
                  onPressed: () => Navigator.pop(context),
                ),
                const SizedBox(height: 8),
                TextButton.icon(
                  icon: const Icon(Icons.logout, color: Colors.red),
                  label: Text(urdu ? 'لاگ آؤٹ' : 'Logout',
                      style: const TextStyle(color: Colors.red)),
                  onPressed: () async {
                    await HistoryDb.clearSession();
                    if (!context.mounted) return;
                    Navigator.pushAndRemoveUntil(
                      context,
                      MaterialPageRoute(
                          builder: (_) => const LoginScreen()),
                      (_) => false,
                    );
                  },
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class HistoryScreen extends StatelessWidget {
  final bool urdu;
  const HistoryScreen({super.key, this.urdu = false});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          tooltip: 'Back',
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(urdu ? 'پچھلی تشخیص' : 'History'),
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
      ),
      body: FutureBuilder<List<Map<String, Object?>>>(
        future: HistoryDb.all(),
        builder: (context, snap) {
          if (!snap.hasData) return const Center(child: CircularProgressIndicator());
          final rows = snap.data!;
          if (rows.isEmpty) return Center(child: Text(urdu ? 'ابھی تک کوئی تشخیص نہیں ہوئی۔' : 'No diagnoses yet.'));
          return ListView.builder(
            itemCount: rows.length,
            itemBuilder: (_, i) {
              final r = rows[i];
              final conf = ((r['confidence'] as double) * 100).toStringAsFixed(1);
              return ListTile(
                leading: const Icon(Icons.eco, color: Colors.green),
                title: Text('${displayLabel('${r['label']}', urdu)}  ($conf%)'),
                subtitle: Text('${r['diagnosed_at']}'),
                trailing: Icon(
                  (r['synced'] as int) == 1 ? Icons.cloud_done : Icons.cloud_off,
                  color: (r['synced'] as int) == 1 ? Colors.green : Colors.grey,
                ),
              );
            },
          );
        },
      ),
    );
  }
}


// ---------------------------------------------------------------------------
// SCAN LINE OVERLAY - a green scanning line that sweeps over the leaf
// photo while the model is working, so scanning FEELS like scanning.
// ---------------------------------------------------------------------------
class ScanLineOverlay extends StatefulWidget {
  const ScanLineOverlay({super.key});
  @override
  State<ScanLineOverlay> createState() => _ScanLineOverlayState();
}

class _ScanLineOverlayState extends State<ScanLineOverlay>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 1300))
    ..repeat(reverse: true);

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _c,
      builder: (_, __) => CustomPaint(
        painter: _ScanLinePainter(_c.value),
        size: Size.infinite,
      ),
    );
  }
}

class _ScanLinePainter extends CustomPainter {
  final double t;
  const _ScanLinePainter(this.t);
  @override
  void paint(Canvas canvas, Size size) {
    final y = t * size.height;
    final glow = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          Colors.greenAccent.withOpacity(0),
          Colors.greenAccent.withOpacity(0.35),
          Colors.greenAccent.withOpacity(0),
        ],
      ).createShader(Rect.fromLTWH(0, y - 26, size.width, 52));
    canvas.drawRect(Rect.fromLTWH(0, y - 26, size.width, 52), glow);
    final line = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 3;
    canvas.drawLine(Offset(0, y), Offset(size.width, y), line);
  }

  @override
  bool shouldRepaint(_ScanLinePainter old) => old.t != t;
}

// ---------------------------------------------------------------------------
// SCANNER CAMERA - a scanner-style camera: live preview with a leaf
// frame, corner brackets and a moving scan line, instead of the plain
// camera app. If the scanner camera cannot start on a phone (camera
// permission not given yet, no camera, any error), it quietly falls
// back to the normal camera (image_picker), so the user always gets
// a working camera.
// ---------------------------------------------------------------------------
class ScannerScreen extends StatefulWidget {
  final bool urdu;
  const ScannerScreen({super.key, this.urdu = false});
  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen>
    with SingleTickerProviderStateMixin {
  cam.CameraController? _controller;
  bool _ready = false;
  bool _flashOn = false;
  bool _busy = false;
  late final AnimationController _anim = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 1600))
    ..repeat(reverse: true);

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final cams = await cam.availableCameras();
      if (cams.isEmpty) throw Exception('no camera');
      final back = cams.firstWhere(
          (c) => c.lensDirection == cam.CameraLensDirection.back,
          orElse: () => cams.first);
      final ctrl = cam.CameraController(back, cam.ResolutionPreset.high,
          enableAudio: false);
      await ctrl.initialize();
      if (!mounted) {
        await ctrl.dispose();
        return;
      }
      _controller = ctrl;
      setState(() => _ready = true);
    } catch (_) {
      _fallback();
    }
  }

  Future<void> _fallback() async {
    if (!mounted) return;
    final picked = await ImagePicker()
        .pickImage(source: ImageSource.camera, imageQuality: 90);
    if (!mounted) return;
    Navigator.pop(context, picked == null ? null : File(picked.path));
  }

  Future<void> _capture() async {
    final ctrl = _controller;
    if (ctrl == null || _busy) return;
    setState(() => _busy = true);
    try {
      final shot = await ctrl.takePicture();
      if (!mounted) return;
      Navigator.pop(context, await _framedPhoto(File(shot.path)));
    } catch (_) {
      if (mounted) setState(() => _busy = false);
    }
  }

  /// The preview frame shows only the middle of the camera picture,
  /// but takePicture() saves the WHOLE frame (with all the background
  /// around the leaf). Crop the centre square - the part the user
  /// actually framed - and fix the file's rotation flag, so the model
  /// sees the leaf big and upright, like the training photos.
  Future<File> _framedPhoto(File raw) async {
    try {
      final bytes = await raw.readAsBytes();
      final dec = img.decodeImage(bytes);
      if (dec == null) return raw;
      final upright = img.bakeOrientation(dec);
      final side =
          upright.width < upright.height ? upright.width : upright.height;
      final cropped = img.copyCrop(upright,
          x: (upright.width - side) ~/ 2,
          y: (upright.height - side) ~/ 2,
          width: side,
          height: side);
      final out = File('${raw.path}_framed.jpg');
      await out.writeAsBytes(img.encodeJpg(cropped, quality: 92));
      return out;
    } catch (_) {
      return raw;
    }
  }

  Future<void> _toggleFlash() async {
    final ctrl = _controller;
    if (ctrl == null) return;
    try {
      await ctrl
          .setFlashMode(_flashOn ? cam.FlashMode.off : cam.FlashMode.torch);
      setState(() => _flashOn = !_flashOn);
    } catch (_) {}
  }

  Future<void> _gallery() async {
    final picked = await ImagePicker()
        .pickImage(source: ImageSource.gallery, imageQuality: 90);
    if (!mounted) return;
    Navigator.pop(context, picked == null ? null : File(picked.path));
  }

  @override
  void dispose() {
    _anim.dispose();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(widget.urdu ? 'پتا اسکینر' : 'Leaf Scanner'),
      ),
      body: !_ready
          ? const Center(
              child: CircularProgressIndicator(color: Colors.greenAccent))
          : Stack(
              fit: StackFit.expand,
              children: [
                cam.CameraPreview(_controller!),
                AnimatedBuilder(
                  animation: _anim,
                  builder: (_, __) => CustomPaint(
                    painter: _ScannerFramePainter(_anim.value),
                    size: Size.infinite,
                  ),
                ),
                Positioned(
                  top: 18,
                  left: 24,
                  right: 24,
                  child: Text(
                    widget.urdu
                        ? 'ایک پتا فریم کے اندر رکھیں'
                        : 'Place one leaf inside the frame',
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        shadows: [Shadow(color: Colors.black, blurRadius: 6)]),
                  ),
                ),
                Positioned(
                  bottom: 28,
                  left: 0,
                  right: 0,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      IconButton(
                        iconSize: 30,
                        color: Colors.white,
                        icon: const Icon(Icons.photo_library),
                        onPressed: _gallery,
                      ),
                      GestureDetector(
                        onTap: _capture,
                        child: Container(
                          width: 74,
                          height: 74,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border:
                                Border.all(color: Colors.greenAccent, width: 4),
                            color: Colors.white24,
                          ),
                          child: _busy
                              ? const Padding(
                                  padding: EdgeInsets.all(20),
                                  child: CircularProgressIndicator(
                                      color: Colors.white),
                                )
                              : const Icon(Icons.camera_alt,
                                  color: Colors.white, size: 34),
                        ),
                      ),
                      IconButton(
                        iconSize: 30,
                        color: _flashOn ? Colors.yellow : Colors.white,
                        icon: Icon(_flashOn ? Icons.flash_on : Icons.flash_off),
                        onPressed: _toggleFlash,
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }
}

class _ScannerFramePainter extends CustomPainter {
  final double t;
  const _ScannerFramePainter(this.t);
  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width * 0.78;
    final h = w; // square frame
    final left = (size.width - w) / 2;
    final top = (size.height - h) / 2 - 30;
    final rect = Rect.fromLTWH(left, top, w, h);

    // Darken everything outside the frame.
    final dim = Paint()..color = Colors.black.withOpacity(0.45);
    final path = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addRRect(RRect.fromRectAndRadius(rect, const Radius.circular(18)));
    path.fillType = PathFillType.evenOdd;
    canvas.drawPath(path, dim);

    // Green corner brackets.
    final corner = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    const len = 34.0;
    canvas.drawLine(rect.topLeft, rect.topLeft + const Offset(len, 0), corner);
    canvas.drawLine(rect.topLeft, rect.topLeft + const Offset(0, len), corner);
    canvas.drawLine(rect.topRight, rect.topRight + const Offset(-len, 0), corner);
    canvas.drawLine(rect.topRight, rect.topRight + const Offset(0, len), corner);
    canvas.drawLine(
        rect.bottomLeft, rect.bottomLeft + const Offset(len, 0), corner);
    canvas.drawLine(
        rect.bottomLeft, rect.bottomLeft + const Offset(0, -len), corner);
    canvas.drawLine(
        rect.bottomRight, rect.bottomRight + const Offset(-len, 0), corner);
    canvas.drawLine(
        rect.bottomRight, rect.bottomRight + const Offset(0, -len), corner);

    // Moving scan line inside the frame.
    final y = top + t * h;
    final glow = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          Colors.greenAccent.withOpacity(0),
          Colors.greenAccent.withOpacity(0.30),
          Colors.greenAccent.withOpacity(0),
        ],
      ).createShader(Rect.fromLTWH(left, y - 22, w, 44));
    canvas.drawRect(Rect.fromLTWH(left, y - 22, w, 44), glow);
    final line = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 3;
    canvas.drawLine(Offset(left, y), Offset(left + w, y), line);
  }

  @override
  bool shouldRepaint(_ScannerFramePainter old) => old.t != t;
}
