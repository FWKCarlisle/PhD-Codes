import imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt


def get_image(image_path):
    image = Image.open(image_path)
    return image

def get_ring_points(center, radius, num_points=720, invert:bool=False):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    if invert:
        angles = angles[::-1]
        radius = -radius
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.vstack((x, y)).T, angles

def get_brightness_values(image, points):
    brightness_values = []
    for point in points:
        x, y = int(point[0]), int(point[1])
        brightness = image.getpixel((x, y))
        if isinstance(brightness, tuple):  # If the image is RGB, convert to grayscale
            brightness = np.mean(brightness)
        brightness_values.append(brightness)
    return brightness_values

def plot_results(image, points, center,angles, brightness_values, count=0):
    indices = np.arange(len(points))  # Create an array of point indices

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Plot the image with points
    ax1.imshow(image, cmap='gray')
    scatter = ax1.scatter(points[:, 0], points[:, 1], c=indices, cmap='inferno', s=10, label='Ring Points')
    ax1.scatter(center[0], center[1], color='blue', s=50, label='Center Point')
    fig.colorbar(scatter, ax=ax1, label='Point Index')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Image with Ring Points and Center Point')
    ax1.legend()

    # Plot the brightness profile
    ax2.scatter(angles, brightness_values, c=indices, cmap='inferno', s=10)
    ax2.set_xlabel('Angle (radians)')
    ax2.set_ylabel('Brightness')
    ax2.set_title('Brightness Profile Along the Ring')
    # fig.colorbar(scatter, ax=ax2, label='Point Index')

    plt.tight_layout()
    
def LEED_analysis(image_path, centre_point, radius, invert: bool = False):
    image = get_image(image_path)
    brightness = []
    for rad in radius:
        ring_points,angles = get_ring_points(centre_point, rad, invert=invert)
        
        brightness_values = get_brightness_values(image, ring_points)
        brightness.append(brightness_values)
        plot_results(image, ring_points, centre_point,angles, brightness_values)
    
    plt.show()

    plt.scatter(angles, brightness[0], label='Outer Ring', c='r')
    plt.scatter(angles, brightness[1], label='Inner Ring', c='b')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Brightness')
    plt.title('Brightness Profile Along the Ring')
    plt.legend()
    plt.show()
def main():
    number = 6771
    raw_path = rf"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\CrCl3\mm38550-2\i06-2-{number}.nxs"  # Path to raw data
    # image_path = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\CrCl3\SiC\6771\image_0017.png"  # Path to image
    image_path = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\CrCl3\HOPG\7029\image_0000.png"  # Path to image

    centre_point = (260, 240)  # Centre of the image (x, y)
    radius = [150,200]  # Radius of the ring
    invert = True # Invert the ring points
    LEED_analysis(image_path=image_path, centre_point=centre_point, radius=radius, invert=invert)

    return 0

if __name__ == '__main__':
    main()



