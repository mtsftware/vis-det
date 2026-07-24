import cv2
import time
from ultralytics import YOLO

model = YOLO("best-rk3588.rknn")
# Test için videonu, kamerayı kullanacaksan 0 yazabilirsin
cap = cv2.VideoCapture("car5.mp4") 

id_mapping = {}
next_custom_id = 1
trusted_ids = set()

# Kullanıcının takip etmek için seçeceği özel (temiz) ID
selected_id = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    start = time.time()
    annotated = frame.copy()

    # Senin çalışan parametrelerin (mps ve imgsz çok kritik)
    results = model.track(
        frame,
        classes=[0,1,2,3,4,5,6,7,8,9],
        conf=0.4,
        imgsz=640,
        verbose=False,
        device="mps", 
        persist=True,
        tracker="bytetrack.yaml"
    )

    if len(results[0].boxes) and results[0].boxes.id is not None:
        for box in results[0].boxes:
            if box.id is None: continue 
            
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            raw_track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Güven oranı %60'ı bir kez bile geçerse güvenilirler listesine al
            if conf >= 0.6:
                trusted_ids.add(raw_track_id)

            # Sadece güvenilir listedekileri çiz
            if raw_track_id in trusted_ids:
                
                # Temiz (Sıralı) ID Ataması
                if raw_track_id not in id_mapping:
                    id_mapping[raw_track_id] = next_custom_id
                    next_custom_id += 1
                
                custom_id = id_mapping[raw_track_id]

                # --- KULLANICI SEÇİM FİLTRESİ ---
                # Eğer terminalden bir ID girilmişse ve döngüdeki nesne o değilse, çizimi atla!
                if selected_id is not None and custom_id != selected_id:
                    continue
                # -------------------------------

                # Sınıflandırma ve Renk Ataması
                if cls in [0, 1]:
                    color = (255, 0, 0) # İnsan için Mavi
                    label = "Human"
                elif cls in [2, 3, 4, 5, 6, 7, 8, 9]:
                    color = (0, 0, 255) # Araç için Kırmızı
                    label = "Vehicle"
                else: 
                    continue
                
                display_text = f"#{custom_id} {label} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, display_text, (x1, max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fps = 1 / (time.time() - start)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("VisDrone Detection & Tracking", annotated)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): 
        break
    elif key == ord("s"):
        try:
            # Ekranda custom_id'ler yazdığı için kullanıcıdan da onu istiyoruz
            val = int(input("\nTakip edilecek ID numarasini girin (Tümünü görmek için 0): "))
            selected_id = None if val == 0 else val
            print(f"Hedef kilitlendi: #{selected_id}")
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")

cap.release()
cv2.destroyAllWindows()