import os
from ultralytics import YOLO
import cv2

videos_dir = '/home/sheded/yolov8/project/videos'

video_path = os.path.join(videos_dir, '1.mp4')

n = len(os.listdir('videos'))+1
output_path = '{}_{}.mp4'.format(video_path.split('.')[0], n)

cap = cv2.VideoCapture('videos/test.mp4')

ret, frame = cap.read()
h, w, _ = frame.shape
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (w, h))

model_path = '/home/sheded/yolov8/project/runs/detect/train2/weights/best.pt'
model= YOLO(model_path)

conf = 0.4
class_name_dict = {0: 'ball'}

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > conf:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        writer.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1)==27:
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()




