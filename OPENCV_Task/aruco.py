"""Detect 4x4, 5x5, and 6x6 ArUco markers and show their pose."""

import cv2
import numpy as np

# Change these settings to match your webcam and printed marker.
CAMERA_NUMBER = 0
MARKER_SIZE = 0.05  # metres: 5 cm
FIELD_OF_VIEW = 70  # estimated camera field of view; calibration is more accurate

if not hasattr(cv2, "aruco"):
    print("Install OpenCV with: python -m pip install opencv-contrib-python numpy")
    raise SystemExit

# Open the webcam and use its actual image size.
camera = cv2.VideoCapture(CAMERA_NUMBER)
if not camera.isOpened():
    print("Could not open webcam. Try changing CAMERA_NUMBER to 1.")
    raise SystemExit

width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
focal = width / (2 * np.tan(np.deg2rad(FIELD_OF_VIEW) / 2))
camera_matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
distortion = np.zeros((5, 1))  # replace with calibration values for accurate depth

# The marker's four 3D corners, ordered to match OpenCV's detected corners.
half = MARKER_SIZE / 2
marker_points = np.array([[-half, half, 0], [half, half, 0],
                          [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

# Make one detector for each marker family.
detectors = []
for name, dictionary_id in [("4x4", cv2.aruco.DICT_4X4_50),
                            ("5x5", cv2.aruco.DICT_5X5_50),
                            ("6x6", cv2.aruco.DICT_6X6_50)]:
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detectors.append((name, cv2.aruco.ArucoDetector(
        dictionary, cv2.aruco.DetectorParameters())))

print("Point the webcam at a marker. Press q or Esc to quit.")
while True:
    success, frame = camera.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for name, detector in detectors:
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue

        for i in range(len(ids)):
            image_points = corners[i].reshape(4, 2).astype(np.float32)
            found, rvec, tvec = cv2.solvePnP(
                marker_points, image_points, camera_matrix, distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            cv2.polylines(frame, [image_points.astype(int)], True, (0, 255, 0), 2)

            if found:
                cv2.drawFrameAxes(frame, camera_matrix, distortion, rvec, tvec,
                                  MARKER_SIZE * 0.65, 2)
                x, y, z = tvec.flatten()
                label = f"{name} ID {ids[i][0]}  X:{x:.2f} Y:{y:.2f} Z:{z:.2f}m"
                cv2.putText(frame, label, tuple(image_points[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Depth: {z:.2f} m", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("ArUco pose - q to quit", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

camera.release()
cv2.destroyAllWindows()
