from ultralytics import YOLO
import cv2


def main():
    # Запуск мобелі
    yolo_model = YOLO('yolov8m.pt')
    video_path = 'test_video2.mp4'

    # Зчитування відео
    cap = cv2.VideoCapture(video_path)

    # Обробка фреймів відео
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Виявлення об'єктів
        results = yolo_model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = box.conf[0].item()
                if conf >= 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0].item())
                    detections.append(((x1, y1, x2 - x1, y2 - y1), conf, cls_id))
        # Візуалізація результатів
        for (box, confidence, class_id, index) in detections:
            x, y, w, h = box
            label = f'{yolo_model.names[class_id]} @ {confidence:.1f}'
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('YOLOv8 Object Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

