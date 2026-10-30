import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO("yolo26n-pose.pt")
cap = cv2.VideoCapture("SV.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30
out = cv2.VideoWriter(
    "SV_skeleton.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)
connections = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6)
]
while True:

    ret, frame = cap.read()

    if not ret:
        break
    results = model(
        frame,
        conf=0.25,
        verbose=False
    )
    black_frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )
    keypoints = results[0].keypoints.xy

    if results[0].keypoints.conf is not None:
        confidences = results[0].keypoints.conf
    else:
        confidences = None
    for person_index, person in enumerate(keypoints):

        points = person.cpu().numpy()

        if confidences is not None:
            scores = confidences[person_index].cpu().numpy()
        else:
            scores = np.ones(len(points))
        for start, end in connections:

            if scores[start] > 0.25 and scores[end] > 0.25:

                x1, y1 = points[start]
                x2, y2 = points[end]

                cv2.line(
                    black_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    3
                )
        for i,(x, y) in enumerate(points):

            if scores[i] > 0.25:

                cv2.circle(
                    black_frame,
                    (int(x), int(y)),
                    5,
                    (0, 0, 255),
                    -1
                )
    cv2.imshow(
        "pose detection",
        black_frame
    )
    out.write(black_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
print("Skeleton video completed!")