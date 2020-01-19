import cv2
from timeit import default_timer as timer
from sort import *

camera = cv2.VideoCapture("pedestrians.avi")
camera.open("pedestrians.avi")
car_cascade = cv2.CascadeClassifier('cascade3.xml')
frame_count = 0
# create instance of SORT
mot_tracker = Sort()
while True:
    start = timer()
    (grabbed, frame) = camera.read()
    grayvideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(grayvideo, 1.1, minNeighbors=4)  # 1.1 #2

    mot_before_list = []
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        mot_before_list.append([x, y, (x+w), (y+h), 0.8])
    mot_before_np = np.array(mot_before_list)
    track_bbs_ids = mot_tracker.update(mot_before_np)
    mot_x1, mot_y1, mot_x2, mot_y2, obj_id = [0, 0, 0, 0, 0]
    trk_with_wid_hgt = []
    for i in track_bbs_ids:
        mot_x1, mot_y1, mot_x2, mot_y2, obj_id = i
        x1_i = int(mot_x1)
        y1_i = int(mot_y1)
        x2_i = int(mot_x2)
        y2_i = int(mot_y2)
        cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (255, 0, 0), 2)
        text = str(obj_id)
        print(text)
        cv2.putText(frame, text, (x1_i, y1_i - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        wid = x2_i - x1_i
        higt = y2_i - y1_i
        trk_with_wid_hgt.append([x1_i, y1_i, wid, higt, obj_id])
    frame_count = frame_count + 1

    cv2.imshow("video", frame)
    end = timer()
    print(int(1 / (end - start)))

    if cv2.waitKey(10) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
