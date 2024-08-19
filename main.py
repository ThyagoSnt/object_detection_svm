import cv2
import numpy as np

def create_mask(hsv_frame):
    # Define the color ranges for blue
    lower_blue = np.array([100, 120, 150])
    upper_blue = np.array([113, 255, 255])
    # Define the color ranges for yellow
    lower_yellow = np.array([20, 50, 100])
    upper_yellow = np.array([40, 140, 255])
    # Create the mask
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    combined_mask = cv2.bitwise_or(blue_mask, yellow_mask)
    return combined_mask

# Function to compute Intersection over Union (IoU) between two bounding boxes
def compute_iou(box1, box2):
    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2, _ = box2

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def non_max_suppression(boxes, threshold):
    if len(boxes) == 0:
        return []
    # Sort boxes based on their area in descending order
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    suppressed = []
    while len(boxes) > 0:
        current_box = boxes.pop(0)
        suppressed.append(current_box)
        boxes = [box for box in boxes if compute_iou(current_box, box) < threshold]
    return suppressed

# Function to apply erosion followed by dilation for cleaning the mask
def clean(image, iterations1=1, iterations2=1):
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((15, 15), np.uint8)
    eroded = cv2.erode(image, kernel1, iterations=iterations1)
    dilated = cv2.dilate(eroded, kernel2, iterations=iterations2)
    return dilated

def find_bases(frame):
    frame = cv2.resize(frame, (640, 480))
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create the mask for blue and yellow colors
    mask = create_mask(hsv)
    mask = clean(mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store the bounding boxes]
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > 2000:
            boxes.append([x, y, w, h, area])

    # Apply Non-Maximum Suppression
    boxes = non_max_suppression(boxes, threshold=0.005)

    if boxes:
        # Draw the bounding boxes on the frame
        for box in boxes:
            x, y, w, h, _ = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame, mask, boxes

def main():
    # Capture the video
    video_path = 'bases.mp4'  # Replace with the path to your video
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        result, mask, boxes = find_bases(frame)

        # Display the result with bounding boxes and the mask
        cv2.imshow('Video with Bounding Boxes', result)
        cv2.imshow('Mask', mask)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()