import matplotlib.pyplot as plt
import numpy as np
from nexusformat.nexus import nxload
import imageio
from PIL import Image, ImageDraw, ImageFont
 
# Opening the test image, and saving it's object


number = 6727 #next
# type = "HOPG" 
type = "SiC"
path = rf"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\CrCl3\mm38550-2\i06-2-{number}.nxs"
save_path = rf"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\CrCl3\{type}\{number}"



def grab_images(path):
    nexus = nxload(path)
    print(nexus.tree)
    images = nexus['/entry/medipix/data']
    return images

def calculate_average_brightness(image):
    return np.mean(image)


def save_images(images, save_path=rf"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\CrCl3\HOPG"):

    slab_size = 100  # Adjust this size based on your memory capacity
    brightneses = []  
# Process images in slabs
    num_images = images.shape[0]
    for start in range(0, num_images, slab_size):
        end = min(start + slab_size, num_images)
        slab = images[start:end]
        print(f'Processing images {start} to {end}')
        for i, image in enumerate(slab):

            image_index = start + i

            # image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image)))
            # image = image.astype(np.uint8)        

            pil_image = Image.fromarray(image)

            # Create a new image with extra space at the bottom for the text
            new_image = Image.new('RGB', (pil_image.width, pil_image.height + 64), (255, 255, 255))
            new_image.paste(pil_image, (0, 0))

            brightness = calculate_average_brightness(new_image)
            brightneses.append(brightness)

            # print("brightness: ",brightness)
            # Draw the text box
            draw = ImageDraw.Draw(new_image)
            font = ImageFont.load_default()
            text = f"Image {image_index}, brightness: {brightness:.2f}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_position = ((new_image.width - text_width) // 2, pil_image.height + (50 - text_height) // 2)
            draw.text(text_position, text, fill="black", font=font)


            imageio.imwrite(rf'{save_path}\image_{image_index:04d}.png', new_image)
        # if i == 99:
        #     plt.imshow(new_image, cmap='gray')
        #     plt.show()
    return brightneses


brightnes = save_images(grab_images(path),save_path)
plt.plot(np.arange(len(brightnes)), brightnes)
plt.xlabel("Image number")
plt.ylabel("Brightness")
plt.show()
print("finished making images")