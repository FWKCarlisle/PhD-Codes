import cv2
import numpy as np

# Load the image

def find_islands(image, show_img = False):
# Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)  # Adjust if needed

    # Remove small noise using morphological operations (optional)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small islands (e.g., noise)
    min_area = 50  # Adjust based on your image
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Sort contours by area (largest first)
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

    # Remove the largest contour (assumed to be the background)
    if len(filtered_contours) > 1:
        filtered_contours = filtered_contours[1:]  # Keep all but the largest

    # Define alternating colors for filling
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0)]  # Red, Green, Blue, Yellow, Orange

    # Compute areas and label islands
    areas = []
    for i, cnt in enumerate(filtered_contours):
        area = cv2.contourArea(cnt)
        areas.append(area)
        print(f"Island {i+1}: Area = {area:.2f} pixels")  # Print each island's area

        # Fill the island with an alternating color
        cv2.drawContours(image, [cnt], -1, colors[i % len(colors)], thickness=cv2.FILLED)

        # Compute center of the island for labeling
        M = cv2.moments(cnt)
        if M["m00"] != 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Put label on the image
            cv2.putText(image, f"{area:.0f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)  # White text for visibility

    # Compute and print average island area
    if areas:
        avg_area = sum(areas) / len(areas)
        print(f"\nAverage Island Area: {avg_area:.2f} pixels")
        num_islands = len(areas)
    else:
        print("\nNo valid islands detected.")
        num_islands = 0

    # Show the image with filled islands
    if show_img:
        cv2.imshow("Islands with Alternating Colors", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the output image
    print(type(image))
    # cv2.imwrite("colored_islands.png", image)
    return image, avg_area, num_islands # Return the image and average area

number = "0800"
image = cv2.imread(rf"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\CrCl3\HOPG\7054\gif_images\image_{number}.png")

find_islands(image, show_img = True)