import cv2
import numpy as np
import matplotlib.pyplot as plt

import json

import ipdb


# Load video
video_path = './tarantino1.mp4'
out_path = './output.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

# Load the background image (extracted from different script)
background_frame = cv2.imread("background.png")

if background_frame is None:
    print("Error: Could not load 'background.png'. Ensure the file exists.")
    exit()

# Resize background
background_frame = cv2.resize(background_frame, (frame_width, frame_height))

# Define refined HSV color ranges for red ball segmentation masks
lower_red1 = np.array([0, 99, 191])
upper_red1 = np.array([20, 199, 255])
lower_red2 = np.array([170, 99, 191])  
upper_red2 = np.array([180, 199, 255])

# Extract a frame to manually adjust color bounds (for HSV)
extract_frame_number = 100  # Adjust as needed
frame_extracted = False

# Load the extracted ball template (for template matching - template exctracted from different script)
template = cv2.imread('ball_template.jpg', 0)  # Load as grayscale
if template is None:
    print("Error: Could not load ball template. Ensure 'ball_template.jpg' exists.")
    exit()
    
w, h = template.shape[::-1]  # Get template dimensions

# Load a smoke effect image (for the carte blanche part)
smoke_img = cv2.imread("smoke.png", cv2.IMREAD_UNCHANGED)

if smoke_img is None:
    print("Error: Could not load 'smoke.png'. Ensure the file exists.")
    exit()

smoke_img = cv2.resize(smoke_img, (frame_width, frame_height))

# Ensure smoke_img has the same number of channels as frames
if smoke_img.shape[2] == 4:  # If it has an alpha channel (RGBA)
    b, g, r, alpha = cv2.split(smoke_img)  # Extract channels
    smoke_img = cv2.merge((b, g, r))  # Convert to BGR (drop alpha)

elif smoke_img.shape[2] != 3:  # Ensure it has 3 channels (BGR)
    smoke_img = cv2.cvtColor(smoke_img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR if needed

# Load the pizza image (for carte blanche part)
pizza_img = cv2.imread("pizza.png", cv2.IMREAD_UNCHANGED)

if pizza_img is None:
    print("Error: Could not load 'pizza.png'. Ensure the file exists.")
    exit()

while True:
    subtitle2 = None # For some subtitles that we need more text
    subtitle3 = None # For some subtitles that shouldbe black text
    
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames

    # Save a frame for color adjustment
    if frame_count == extract_frame_number and not frame_extracted:
        cv2.imwrite("color_frame.jpg", frame)
        frame_extracted = True

    # Switch between grayscale and original every second
    if frame_count < 4 * fps:
        if frame_count // fps % 2 == 0:
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)  # Convert back for writing
        else:
            processed_frame = frame
        subtitle = "Gray scale: Switching between gray scale and original."

    # Gaussian filtering
    elif frame_count < 8 * fps:
        kernel_size = (5 + (frame_count - 4 * fps) // (fps // 4) * 2, 
                       5 + (frame_count - 4 * fps) // (fps // 4) * 2)
        processed_frame = cv2.GaussianBlur(frame, kernel_size, 0)
        subtitle = "Increasing Gaussian Blur: Blurs edges, reduces noise."

    # Bilateral filtering
    elif frame_count < 12 * fps:
        diameter = 9 + (frame_count - 8 * fps) // (fps // 4) * 4
        processed_frame = cv2.bilateralFilter(frame, diameter, 75, 75)
        subtitle = "Increasing Bilateral Filter: Preserves edges while smoothing."

    # RGB-based object grabbing
    elif frame_count < 14 * fps:
        b, g, r = cv2.split(frame)
        mask_rgb = cv2.inRange(b, 100, 255)

        # Invert the mask so the foreground is white and background is black
        mask_rgb = cv2.bitwise_not(mask_rgb)    

        processed_frame = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2BGR)
        subtitle3 = "RGB Segmentation: Grabbing object in one channel."

    # HSV-based segmentation
    elif frame_count < 18 * fps:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_hsv = cv2.bitwise_or(mask1, mask2)
        processed_frame = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
        subtitle = "HSV Segmentation: More robust to lighting variations."
    
    # Morphological operations for optimized grabbing
    elif frame_count < 22 * fps:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_hsv = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((9, 9), np.uint8)
        mask_hsv_refined = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise
        mask_hsv_refined = cv2.morphologyEx(mask_hsv_refined, cv2.MORPH_CLOSE, kernel, iterations=3)  # Fill holes

        # Compute difference (show only improvements)
        mask_improvements = cv2.subtract(mask_hsv_refined, mask_hsv)
        mask_improvements_colored = cv2.cvtColor(mask_improvements, cv2.COLOR_GRAY2BGR)
        mask_improvements_colored[:, :, 0] = 255  # Highlight improvements in blue
        # mask_improvements_colored[mask_improvements == 0] = 0

        # Overlay improvements on the refined mask
        processed_frame = cv2.cvtColor(mask_hsv_refined, cv2.COLOR_GRAY2BGR)
        processed_frame = cv2.addWeighted(processed_frame, 1, mask_improvements_colored, 0.5, 0)

        subtitle = "Morphological Processing: Blue = Fixed Segmentation Areas"

        subtitle2 = "Note: HSV segmentation was already almost perfect in our case"

    # Sobel edge detector
    elif frame_count < 27 * fps:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dynamically adjust Sobel parameters
        if frame_count < 25 * fps:  
            ksize, scale_factor, delta_value, ddepth = 1, 1, 0, cv2.CV_64F  # Default settings
        elif frame_count < 26 * fps:  
            ksize, scale_factor, delta_value, ddepth = 3, 1, 0, cv2.CV_64F  # More intense edges
        else:  
            ksize, scale_factor, delta_value, ddepth = 3, 2, 0, cv2.CV_64F  # Strong contrast edges

        # Compute Sobel edges with dynamic parameters
        sobel_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=ksize, scale=scale_factor, delta=delta_value)
        sobel_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=ksize, scale=scale_factor, delta=delta_value)

        # Convert to absolute values
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)

        # Enhance contrast
        enhanced_x = cv2.addWeighted(abs_sobel_x, 1.5, abs_sobel_y, 0, 0)
        enhanced_y = cv2.addWeighted(abs_sobel_y, 1.5, abs_sobel_x, 0, 0)

        # Merge into a colorful output
        processed_frame = cv2.merge([enhanced_x, enhanced_y, abs_sobel_x + abs_sobel_y])

        subtitle = f"Sobel Edge Detection: ksize={ksize}, scale={scale_factor}, delta={delta_value}, depth={ddepth}."

    # Hough Circle Transform
    elif frame_count < 38 * fps:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 5)

        # Dynamic parameter changes over time (blue: weak detection, yellow: medium detection, green: strict deetction)
        if frame_count < 30 * fps:
            param1, param2, minRadius, maxRadius = 30, 20, 5, 60
            circle_color = (255, 0, 0)
        elif frame_count < 34 * fps:
            param1, param2, minRadius, maxRadius = 45, 35, 7, 55
            circle_color = (0, 255, 255)
        else:
            param1, param2, minRadius, maxRadius = 50, 30, 10, 50
            circle_color = (0, 255, 0)

        # Hough Circle Transform with dynamic parameters
        circles = cv2.HoughCircles(
            gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], circle_color, 2)  # Dynamic color based on phase
                cv2.circle(frame, (i[0], i[1]), 2, (255, 255, 0), 3)  # Yellow dot at center

        processed_frame = frame
        subtitle = f"Hough Circle Detection: param1={param1}, param2={param2}, minR={minRadius}, maxR={maxRadius}"

    # Template Matching
    elif frame_count < 40 * fps:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.5:  # Ensure detection confidence is high
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # Flash color every 10 frames (adjust for faster/slower flashing)
            if (frame_count // 10) % 2 == 0:
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 0, 0)  # Blue

            cv2.rectangle(frame, top_left, bottom_right, color, 3)
            subtitle = "Template Matching: Highlighting detected object."

        processed_frame = frame.copy()

    # Grayscale Likelihood Map
    elif frame_count < 43 * fps:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # Resize the template if too large
        max_template_size = 100
        if template.shape[0] > max_template_size or template.shape[1] > max_template_size:
            scale_factor = max_template_size / max(template.shape)
            template_resized = cv2.resize(template, (int(template.shape[1] * scale_factor), int(template.shape[0] * scale_factor)))
        else:
            template_resized = template

        # Use cv2.matchTemplate with squared difference
        res = cv2.matchTemplate(gray, template_resized, cv2.TM_SQDIFF_NORMED)

        # Invert heatmap - white = high confidence, black = low confidence
        likelihood_map = 1 - res

        # Contrast enhancement
        likelihood_map = np.power(likelihood_map, 10)

        # Normalize
        likelihood_map = cv2.normalize(likelihood_map, None, 0, 255, cv2.NORM_MINMAX)
        likelihood_map = np.uint8(likelihood_map)

        # Resize likelihood map
        likelihood_map_resized = cv2.resize(likelihood_map, (frame.shape[1], frame.shape[0]))

        # Apply a threshold to ensure weak matches stay dark
        _, thresholded_map = cv2.threshold(likelihood_map_resized, 150, 255, cv2.THRESH_TOZERO)

        processed_frame = cv2.cvtColor(likelihood_map_resized, cv2.COLOR_GRAY2BGR)

        subtitle = "Likelihood Map: White = High certainty, Black = Low certainty."    

    # Carte blanche 1: Color changing effect
    elif 44 * fps < frame_count < 50 * fps:
        # Colors: Yellow, Green, Blue, Pink
        yellow = (0, 255, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        pink = (255, 192, 203)

        # HSV mask
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_hsv = cv2.bitwise_or(mask1, mask2)

        frame_copy = frame.copy()

        if (frame_count // fps) % 4 == 0:
            color = pink
        if (frame_count // fps) % 4 == 1:
            color = yellow
        elif (frame_count // fps) % 4 == 2:
            color = green
        elif (frame_count // fps) % 4 == 3:
            color = blue

        if color:
            object_only = np.zeros_like(frame)
            object_only[mask_hsv > 0] = color  # Apply color only to detected object
            frame_copy = cv2.addWeighted(frame, 0.6, object_only, 0.4, 0)  # Blend colors smoothly

        subtitle = "Carte Blanche 1: Color Changing Effect"
        processed_frame = frame_copy

    # Carte blanche 2: Object dissapear into smoke
    elif 51 * fps < frame_count < 55 * fps:
        # Absolute difference between current frame and background
        diff = cv2.absdiff(background_frame, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Apply an adaptive threshold to create a more accurate mask
        mask = cv2.adaptiveThreshold(gray_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        # Apply morphological operations to refine the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_inv = cv2.bitwise_not(mask)

        # Extract the backgrounds in current frame and background image
        background_region = cv2.bitwise_and(background_frame, background_frame, mask=mask)
        current_background = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Merge the background and the current frame to make the object disappear
        processed_frame = cv2.add(background_region, current_background)

        # Text on image
        alpha_text = (frame_count - 51 * fps) / (55 * fps - 51 * fps)
        poof_color = (255, 255, 255)  # White text
        cv2.putText(processed_frame, "Poof!", (frame_width // 2 - 50, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, poof_color, 5, cv2.LINE_AA)

        # Ensure both images have the same size
        processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))
        smoke_img = cv2.resize(smoke_img, (frame_width, frame_height))

        # Smoke effect - apply gradually smoke.png
        alpha_smoke = (frame_count - 51 * fps) / (55 * fps - 51 * fps)
        alpha_smoke = max(0.1, min(alpha_smoke, 1))
        smoke_overlay = cv2.addWeighted(smoke_img, alpha_smoke, processed_frame, 1, 0)
        processed_frame = smoke_overlay

        subtitle = "Carte Blanche 2: Object disappearing into smoke"

    # Carte blanche 3: Replace ball with pizza
    elif 55 * fps < frame_count <= 60 * fps:
        # HSV for object detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create the mask
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_hsv = cv2.bitwise_or(mask1, mask2)

        # Find contours for ball's bounding box
        contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest detected contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Scale up pizza size
            scale_factor = 1.2
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)

            #Ensure pizza stays within the frame
            new_x = max(0, x - (new_w - w) // 2)
            new_y = max(0, y - (new_h - h) // 2)
            new_w = min(frame_width - new_x, new_w)
            new_h = min(frame_height - new_y, new_h)

            # Resize the pizza
            pizza_resized = cv2.resize(pizza_img, (new_w, new_h))

            # Ensure pizza has 3 channels (BGR)
            if pizza_resized.shape[2] == 4:
                b, g, r, alpha = cv2.split(pizza_resized)
                pizza_resized = cv2.merge((b, g, r))

            # Create a mask for the pizza
            pizza_gray = cv2.cvtColor(pizza_resized, cv2.COLOR_BGR2GRAY)
            _, pizza_mask = cv2.threshold(pizza_gray, 1, 255, cv2.THRESH_BINARY)
            pizza_mask_inv = cv2.bitwise_not(pizza_mask)

            # Extract the background where the pizza will be placed
            roi = frame[new_y:new_y+new_h, new_x:new_x+new_w]
            background = cv2.bitwise_and(roi, roi, mask=pizza_mask_inv)

            # Place the pizza onto the detected ball position
            pizza_final = cv2.add(background, pizza_resized)
            frame[new_y:new_y+new_h, new_x:new_x+new_w] = pizza_final

        processed_frame = frame

        # Text on image
        alpha_text = (frame_count - 55 * fps) / (60 * fps - 55 * fps)
        poof_color = (255, 255, 255)  # White text
        cv2.putText(processed_frame, "Look, it's a moving pizza!", (frame_width // 2 - 100, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, poof_color, 5, cv2.LINE_AA)
        
        subtitle = "Carte Blanche 3: Ball Replaced with Pizza"

    else:
        processed_frame = frame
        subtitle = " "
        
    # Add subtitles
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(processed_frame, subtitle, (30, frame_height - 80), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    if subtitle2:
         cv2.putText(processed_frame, subtitle2, (30, frame_height - 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    elif subtitle3:
        cv2.putText(processed_frame, subtitle3, (30, frame_height - 80), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the video
    cv2.imshow("Live", processed_frame)

    # Write frame to output video
    out.write(processed_frame)
    
    frame_count += 1
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
      
cv2.destroyAllWindows()
cap.release()
out.release()
print("Video processing complete. Output saved at:", out_path)