"""
This project, including this file, is licensed under MPL-2.0.
"""

from pathlib import Path

import imageio.v2 as imageio
from tkinter.filedialog import askdirectory
from PIL import Image, ImageChops



def make_gif(dirname: Path, duration: int = 50, gif_name: str = "made_gif.gif", invert: bool = False):
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
        for filename in images:
            if invert:
                img = Image.open(filename)
            
                # Passing the image object to invert()  
                inv_img = ImageChops.invert(img)
                
                # Displaying the output image
                image = inv_img
                # image = imageio.imread(inv_img)
            else:
                image = imageio.imread(filename)
            writer.append_data(image)

    return output


def make_gif_options(pop_up=True, directory=None, name="made_gif.gif", invert = True):
    """
    Scripting interface for make_gif. Opens a window asking the user to choose a
    directory, and runs make_gif on that directory.
    """
    if pop_up:
        png_dir = askdirectory(title="Select Folder")
        make_gif(png_dir, invert=invert, gif_name=name)
    elif directory is not None:
        png_dir = directory
        make_gif(png_dir, invert=invert, gif_name=name)
    else:
        raise ValueError("Either pop_up or directory must be have values!")
        
    print(f"finished making gif of name {name}")

if __name__ == "__main__":
    directory = "C:/Users/ppxfc1/OneDrive - The University of Nottingham/Desktop/PhD/CrCl3/HOPG/7050"
    name = "made_gif.mp4"
    make_gif_options(pop_up=False, directory=directory, name=name, invert=False)

