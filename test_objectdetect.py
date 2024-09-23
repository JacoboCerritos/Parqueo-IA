import cv2

import torch

from ultralytics import YOLO


# Initialize the YOLOv8 model

model = YOLO('Test.pt')


# Open the camera

cap = cv2.VideoCapture(1)


while True:

    # Read a frame from the camera

    ret, frame = cap.read()


    # Perform object detection on the frame

    results = model(frame)


    # Draw the bounding boxes and labels on the frame

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0]

        label = results[0].names[int(box.cls[0])]

        confidence = box.conf[0]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


        cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the frame

    cv2.imshow("Object Detection", frame)


    # Exit if the user presses the 'q' key

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


# Release the camera and destroy all windows

cap.release()

cv2.destroyAllWindows()