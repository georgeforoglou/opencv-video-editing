import cv2
import numpy as np
import matplotlib.pyplot as plt

frame_path = "sample_frame.jpg"
frame = cv2.imread(frame_path)

if frame is None:
    print("Error: Could not load the frame. Ensure 'sample_frame.jpg' exists.")
    exit()

frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(frame_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(frame_hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    ball_template = frame[y:y+h, x:x+w]

    template_path = "ball_template.jpg"
    cv2.imwrite(template_path, ball_template)

    frame_with_box = frame.copy()
    cv2.rectangle(frame_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Detected Red Ball (Bounding Box)")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(ball_template, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Extracted Ball Template")
    axes[1].axis("off")

    plt.show()

    print(f"Ball template saved as {template_path}")

else:
    print("Error: Red ball not detected")
