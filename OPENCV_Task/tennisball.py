"""Find blue objects and a tennis ball in a live camera feed."""

import argparse
from collections import deque

import cv2
import numpy as np

# HSV color ranges. Adjust them if your camera or lighting is different.
BLUE_LOW = np.array([90, 70, 40])
BLUE_HIGH = np.array([135, 255, 255])
BALL_LOW = np.array([20, 70, 70])
BALL_HIGH = np.array([45, 255, 255])

blue_objects = []
selected_center = None
trail = deque(maxlen=40)

def find_blue_objects(mask):
    """Return blue regions with enough area, their centers, and boxes."""
    objects = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center = (int(moments["m10"] / moments["m00"]),
                  int(moments["m01"] / moments["m00"]))
        objects.append((contour, area, center, (x, y, width, height)))
    return objects

def find_tennis_ball(mask):
    """Return the largest round yellow-green region, if one is visible."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 150 or perimeter == 0:
            continue

        # A circle has circularity close to 1. Blue objects do not use this test.
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, width, height = cv2.boundingRect(contour)
        if circularity > 0.55 and 0.65 < width / max(height, 1) < 1.5:
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            balls.append(((int(center_x), int(center_y)), int(radius), area))
    return max(balls, key=lambda ball: ball[2]) if balls else None

def select_blue_object(event, x, y, flags, _data):
    """Select a blue object when the user clicks inside its box."""
    global selected_center, trail
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for _contour, _area, center, (left, top, width, height) in blue_objects:
        if left <= x <= left + width and top <= y <= top + height:
            selected_center = center
            trail.clear()
            trail.append(center)
            return

def main():
    global blue_objects, selected_center, trail

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera number (default: 0)")
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")

    window = "Blue Object and Tennis Ball Tracker"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, select_blue_object)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert to HSV so color is easier to separate from brightness.
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
        ball_mask = cv2.inRange(hsv, BALL_LOW, BALL_HIGH)

        # Clean small specks in both color masks.
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        blue_objects = find_blue_objects(blue_mask)
        ball = find_tennis_ball(ball_mask)
        output = frame.copy()

        # Draw and label each blue object. No circular shape is assumed.
        for _contour, area, _center, (x, y, width, height) in blue_objects:
            cv2.rectangle(output, (x, y), (x + width, y + height), (255, 140, 0), 2)
            cv2.putText(output, f"Blue area: {area:.0f} px^2", (x, max(y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2)

        # Follow the selected blue object's nearest center in the next frame.
        if selected_center is not None and blue_objects:
            chosen = min(blue_objects, key=lambda obj: np.linalg.norm(
                np.array(obj[2]) - np.array(selected_center)))
            _contour, _area, selected_center, (x, y, width, height) = chosen
            trail.append(selected_center)
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 3)
            cv2.circle(output, selected_center, 5, (0, 0, 255), -1)
            for start, end in zip(trail, list(trail)[1:]):
                cv2.line(output, start, end, (0, 0, 255), 2)
            cv2.putText(output, f"Selected blue center: {selected_center}",
                        (10, 80 if ball else 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2)

        if ball:
            center, radius, area = ball
            cv2.circle(output, center, radius, (0, 255, 255), 2)
            cv2.circle(output, center, 4, (0, 0, 255), -1)
            cv2.putText(output, f"Ball center: {center}  radius: {radius}px  area: {area:.0f}px^2",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            if selected_center:
                dx = selected_center[0] - center[0]
                dy = selected_center[1] - center[1]
                cv2.line(output, center, selected_center, (255, 0, 255), 2)
                cv2.putText(output, f"Blue relative to ball: dx={dx:+d}px dy={dy:+d}px",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        else:
            cv2.putText(output, "Tennis ball not detected", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.putText(output, "Click blue to select | c: clear | q: quit",
                    (10, output.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.imshow(window, output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            selected_center = None
            trail.clear()

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
