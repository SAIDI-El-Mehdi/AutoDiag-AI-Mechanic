import sys
# كنجيبو المكتبات من الديسك D
sys.path.append(r"D:\tf_files")

from ultralytics import YOLO

# 1. كنجيبو الدماغ الأصلي ديال البيسي
model = YOLO("best.pt")

print("بدينا عملية التحويل للتيليفون... هادشي يقدر ياخد شوية د الوقت...")

# 2. كنحولو الدماغ لصيغة tflite ديال الأندرويد
model.export(format="tflite")

print("مبروك! التحويل سالا بنجاح.")