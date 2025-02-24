"""
This project, including this file, is licensed under MPL-2.0.
"""

from pathlib import Path

import imageio.v2 as imageio
from tkinter.filedialog import askdirectory
from PIL import Image, ImageChops
import numpy as np



def make_movie(dirname: Path, duration: int = 50, gif_name: str = "made_gif.mp4", invert: bool = False, threshold: bool =False) -> Path:
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

    images = sorted(dirname.rglob("*.png"))
    if not images:
        raise RuntimeError(f"No png files found in: {dirname}")

    output = dirname / gif_name
    
   


    with imageio.get_writer(output, mode="I", fps=60) as writer:
        for i, filename in enumerate(images):
            image = Image.open(filename)
            if i == 0:
                    print("Thresholding the image so that only the dark spots are visable.")
                    print("This is done by setting values below a certain threshold to 0.")
                    print("If you do not want this, set threshold = False")

                    background = image

            image_np = np.array(image) / np.max(background)      
            if invert:

                # Passing the image object to invert()  
                inv_img = ImageChops.invert(image)
                image_np = np.array(inv_img)
                # image = imageio.imread(inv_img)
            if threshold:
                
                ### Thresholding the image so that only the dark spots are visable.
                
                threshold_value = 50  # percentage of the maximum value
                max_value = np.max(image_np)
                image_np[image_np < (threshold_value / 100) * max_value] = 0

            else:
                image_np = imageio.imread(filename)
            
            writer.append_data(image_np)

    return output


def make_gif_options( directory=None, name="made_gif.gif", pop_up=True,invert = True):
    """
    Scripting interface for make_gif. Opens a window asking the user to choose a
    directory, and runs make_gif on that directory.
    """
    if pop_up:
        png_dir = askdirectory(title="Select Folder")
        make_movie(png_dir, invert=invert, gif_name=name)
    elif directory is not None:
        png_dir = directory
        make_movie(png_dir, invert=invert, gif_name=name)
    else:
        raise ValueError("Either pop_up or directory must be have values!")
        
    print(f"finished making gif of name {name}")

if __name__ == "__main__":
    directory = "C:/Users/ppxfc1/OneDrive - The University of Nottingham/Desktop/PhD/CrCl3/HOPG/7029"
    name = "made_gif.mp4"
    make_gif_options( directory=directory, name=name, pop_up=False,invert=False)

