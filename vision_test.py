import cv2
import pupil_apriltags as apriltag
import numpy as np 
from ultralytics import YOLO

people = {
    0: 'stella',
    1: 'tycho'
}

def is_point_in_box(px, py, x_min, y_min, x_max, y_max):
    return (x_min <= px <= x_max) and (y_min <= py <= y_max)

# Use 0 for the default built-in webcam, or pass a file path string like 'video.mp4'
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the video source.")
    exit()

detector = apriltag.Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

model = YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Stream ended or frame cannot be fetched. Exiting...")
        break

    person_bb = [] # min x, max x, min y, max y

    results = model(frame, classes=[0], verbose=False) # 0 is person
    for box in results[0].boxes:
        (x1, y1, x2, y2) = box.xyxy.tolist()[0]
        top_left = (int(x1), int(y1))
        bottom_right = (int(x2), int(y2))
        top_right = (int(x2), int(y1))
        bottom_left = (int(x1), int(y2))

        # draw bounding box
        cv2.line(frame, top_left, top_right, (0, 0, 255), 2)
        cv2.line(frame, top_right, bottom_right, (0, 0, 255), 2)
        cv2.line(frame, bottom_right, bottom_left, (0, 0, 255), 2)
        cv2.line(frame, bottom_left, top_left, (0, 0, 255), 2)
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]

        cv2.putText(frame, f"class: {class_name}. confidence {box.conf[0]:.3g}", top_left,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if box.conf[0] > 0.5:
            person_bb = [x1, x2, y1, y2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = detector.detect(gray)

    # loop over the AprilTag detection results
    for r in res:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        tag_id = int(r.tag_id)

        # this doesn't really work if there are multiple people detected in the frame.
        # but tags could be used to identify people like so
        if tag_id in people and person_bb and \
            is_point_in_box(cX, cY, person_bb[0], person_bb[2], person_bb[1], person_bb[3]):
            
            cv2.putText(frame, people[tag_id], (ptA[0], ptA[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Video Playback', frame)

    # Stop the loop if the 'q' key is pressed on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
