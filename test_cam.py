import sys
sys.path.append(r"D:\tf_files")

from ultralytics import YOLO
import cv2

# 1. دخل الدماغ لي صوبنا
model = YOLO("best.pt")

# 2. 💡 قاعدة البيانات ديال الدياغنوستيك: السمية + النصيحة
parts_info = {
    0: {"name": "Oil Cap", "diag": "Check engine oil level"},
    1: {"name": "Battery", "diag": "Check voltage (12.6V) & terminals"},
    2: {"name": "Coolant Reservoir", "diag": "Check for leaks & fluid level"},
    3: {"name": "Fluid Reservoir", "diag": "Check level"},
    4: {"name": "Fuse Box", "diag": "Inspect for blown fuses"},
    6: {"name": "Alternator", "diag": "Check charging system (14V)"},
    7: {"name": "Cap", "diag": "Ensure it is tightly closed"},
    8: {"name": "Fluid Cap", "diag": "Inspect seals"},
    12: {"name": "Air Intake", "diag": "Check air filter condition"},
    13: {"name": "Radiator", "diag": "Inspect fins & cooling fans"},
    15: {"name": "Engine Cover", "diag": "Main Engine Block"}
}

# حل الكاميرا
cap = cv2.VideoCapture(0)
print("الكاميرا شعلات... ورك على حرف 'q' باش تخرج")

while True:
    success, frame = cap.read()
    if not success:
        break

    # عطي الصورة للدماغ
    results = model(frame)

    # 3. 💡 الرسم اليدوي: حنا لي غنكتبو السميات والدياغنوستيك
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # نجيبو بلاصة المربع
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # نجيبو الكود ديال البياسة ونسبة التأكد
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # نقلبو على البياسة فقاعدة البيانات ديالنا
            if cls_id in parts_info:
                part_name = parts_info[cls_id]["name"]
                diagnostic = parts_info[cls_id]["diag"]
            else:
                part_name = f"Unknown Part ({cls_id})"
                diagnostic = "No diagnostic available"

            # نرسمو المربع بالخضر
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # نكتبو السمية د البياسة الفوق (بالزرق)
            cv2.putText(frame, f"{part_name} {conf:.2f}", (x1, y1 - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # نكتبو نصيحة الدياغنوستيك تحت منها (بالصفر)
            cv2.putText(frame, diagnostic, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # بين النتيجة
    cv2.imshow("Auto Diag - Mobile App Prototype", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()