"""
This project, including this file, is licensed under MPL-2.0.
"""

from pathlib import Path
import cv2
import imageio.v2 as imageio
from tkinter.filedialog import askdirectory
from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt

def threshold_image(image, threshold_value: int = 50,background_img = None):
    '''
    image = numpy array of image,
    threshold_value = percentage of the maximum value
    
    '''
      # percentage of the maximum value
    if background_img is not None:
        max_value = np.max(background_img)
    else:
        max_value = np.max(image)
        
    thresh = (threshold_value / 100) * max_value
    image[image < thresh] = 0
    image[image >= thresh] = 255

    return image

def calculate_average_brightness(image):
    return np.mean(image)


def find_islands(image_np, show_img = False):
# Convert to grayscale

    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    gray_eq = cv2.equalizeHist(gray)

    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)  # Adjust if needed, change 180 to the threshold value for brightness
    # _, binary = cv2.threshold(gray_eq, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Remove small noise using morphological operations (optional)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
    perimeters = []
    for i, cnt in enumerate(filtered_contours):
        area = cv2.contourArea(cnt)
        areas.append(area)
        perimeter = cv2.arcLength(cnt, True)
        perimeters.append(perimeter)
        # print(f"Island {i+1}: Area = {area:.2f} pixels")  # Print each island's area

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
        # print(f"\nAverage Island Area: {avg_area:.2f} pixels")
    else:
        print("No valid islands detected.")
        num_islands = 0

    # Show the image with filled islands
    if show_img:
        cv2.imshow("Islands with Alternating Colors", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the output image
    # cv2.imwrite("colored_islands.png", image)
    return image, areas, perimeters # Return the image and average area


def make_movie(dirname: Path, FPS: int = 30, gif_name: str = "made_gif.mp4", invert: bool = False, threshold: int =False, save_images: bool = True, contour: bool = False) -> Path:
    """
    Creates a gif based off a set of given .pngs.  It is important to note that the only
    accepted filetype is .png, although this is planned to be expanded in later updates.

    This functions takes a directory and turns all .pngs within it into a gif with a set
    duration per slide.

    Creates a GIF in the same directory specified for the images.

    Args
    ----
    dirname: Path
        This is the path for the directory which should contain .pngs to be created.
    duration: int, default 2
        This will set the duration for each frame within the gif, in seconds.
    gif_name: str, default "Jorek.gif"
        This is the name of the gif to be added to the directory with the images.

    Returns
    -------
    Path
        Path to the created Gif.

    Raises
    ------
    NotADirectoryError
        If dirname is not a valid directory
    RuntimeError
        If no png files are found.

    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        raise NotADirectoryError(dirname)

    images = sorted(dirname.glob("*.png"))
    if not images:
        raise RuntimeError(f"No png files found in: {dirname}")

    output = dirname / gif_name
    
   
    brighnesses = []
    avg_areas = []
    coverages = []
    compactnesses = []
    first_img = True

    low_bounds = 1
    high_bounds = 2300

    with imageio.get_writer(output, mode="I", fps=FPS) as writer:
        for i, filename in enumerate(images):
            image = Image.open(filename)
            if i == 0:
                    print("Thresholding the image so that only the dark spots are visable.")
                    print("This is done by setting values below a certain threshold to 0.")
                    print("If you do not want this, set threshold = False")


            if i < low_bounds or i > high_bounds:
                continue

            
            image_np = np.array(image)   

            if i > 600 and first_img:
                first_img = False
                background = image_np
            
            else:
                background = None

            
            if invert:

                # Passing the image object to invert()  
                inv_img = ImageChops.invert(image)
                image_np = np.array(inv_img)
                # image = imageio.imread(inv_img)
            if threshold:
                
                image_np = threshold_image(image_np, threshold_value=threshold, background_img=background)
                

            if not(threshold or invert):
                image_np = imageio.imread(filename)

            img_brightness = calculate_average_brightness(image_np)

            brighnesses.append(img_brightness)   
            
            
            image_np = image_np.astype(np.uint8)
            new_image = Image.fromarray(image_np)

            if contour:
                
                contour_img, areas, perimeters = find_islands(image_np, show_img = False)
                if len(areas) > 0:
                    avg_area = sum(areas) / len(areas)
                else:
                    avg_area = 0
                total_area = sum(areas)
                num_islands = len(areas)
                avg_areas.append(avg_area)
            else:
                contour_img = image_np
                avg_area = 0
                total_area = 0
                num_islands = 0
                avg_areas.append(avg_area)
                areas = [1]
                perimeters = [1]


            kit = []
            for j in range(len(areas)):
                AP = areas[j] / perimeters[j]
                kit.append(AP)
            compactness = np.mean(kit)
            compactnesses.append(compactness)

            coverage = total_area / (image_np.shape[0] * image_np.shape[1]) * 100

            coverages.append(coverage)



            writer.append_data(contour_img)
            if save_images:
                # imageio.imwrite(rf'{dirname}\gif_images\image_{i:04d}.png', contour_img)
                # print(rf'{dirname}\gif_images\image_{i:04d}.png')
                cv2.imwrite(rf'{dirname}\gif_images\image_{i:04d}.png', contour_img)

    fig, [ax1, ax2, ax4] = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(np.arange(len(brighnesses)), brighnesses)
    ax1.set_title("Brightness of the image")
    ax1.set_xlabel("Image number")
    ax1.set_ylabel("Brightness")
    
    ax2.plot(np.arange(len(avg_areas)), avg_areas)
    ax2.set_title("Area of the islands")
    ax2.set_xlabel("Image number")
    ax2.set_ylabel("Area")

    # ax3.plot(np.arange(len(coverages)), coverages)
    # ax3.set_title("Coverage of the islands")
    # ax3.set_xlabel("Image number")
    # ax3.set_ylabel("Coverage (%)")

    ax4.plot(np.arange(len(compactnesses)), compactnesses)
    ax4.set_title("Area/Perimeter of the islands")
    ax4.set_xlabel("Image number")
    ax4.set_ylabel("Compactness (Area/Perimeter)")

    plt.show()
    return output


def make_gif_options( directory=None, name="made_gif.gif", pop_up=False,invert = True, threshold = False, save_images = True, contour = False):
    """
    Scripting interface for make_gif. Opens a window asking the user to choose a
    directory, and runs make_gif on that directory.
    """
    if pop_up:
        png_dir = askdirectory(title="Select Folder")
        make_movie(png_dir, invert=invert, gif_name=name, threshold=threshold, save_images=save_images, contour=contour)
    elif directory is not None:
        png_dir = directory
        make_movie(png_dir, invert=invert, gif_name=name, threshold=threshold,save_images=save_images, contour=contour)
    else:
        raise ValueError("Either pop_up or directory must be have values!")
        
    print(f"finished making gif of name {name}")

if __name__ == "__main__":
    # directory = "C:/Users/ppxfc1/OneDrive - The University of Nottingham/Desktop/PhD/CrCl3/SiC/67"
    directory = "C:/Users/ppxfc1/OneDrive - The University of Nottingham/Desktop/PhD/CrCl3/HOPG/7050"

    name = "made_gif.mp4"
    make_gif_options( directory=directory, name=name, pop_up=False,invert=True, threshold=180, save_images=True, contour=True)

