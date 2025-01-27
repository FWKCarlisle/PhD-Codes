import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D


class DBTF_1D:
    def __init__(self, position=[0,0], angle=0):
        self.position = position
        self.angle = angle
        self.inrail = False
        

        self.trans_step_size = 1
        self.rot_step_size = 15
        self.X_diffusion_rate = 1
        self.Y_diffusion_rate = 1
        self.Rotation_diffusion_rate = 0.1


        self.rate_translation_x = self.calculate_rate(self.trans_step_size, self.X_diffusion_rate, 'Translation_X')
        self.rate_translation_y = self.calculate_rate(self.trans_step_size, self.Y_diffusion_rate, 'Translation_Y')
        self.rate_rotation = self.calculate_rate(self.rot_step_size, self.Rotation_diffusion_rate, 'Rotation')

        

        self.total_rate = self.calculate_total_rate()


    def calculate_rate(self, step_size, diffusion_rate, type='Translation_X'):
        '''
        Step size is the size of the lattice
        diffusion_rate is the rate of diffusion of the molecule
        type is the type of motion of the molecule which is randomly selected.
        '''
        if type == "Translation_X" or type == "Translation_Y":
            rate = 2 * diffusion_rate / (step_size ** 2)
        elif type == "Rotation" :
            rate = 2 * diffusion_rate / (step_size ** 2)
        else:
            raise ValueError('Invalid type of motion: Select either "Translation" or "Rotation"')

        return rate

    def find_translations(self):
        '''
        Find the orientations of the molecule
        '''
        translations = []
        if not self.inrail:
            translations.append("Rotation")
        translations.append("Translation_X")
        translations.append("Translation_Y")
        translations.append("Translation_X")
        translations.append("Translation_Y")


        self.orientations = translations

        return translations

    def calculate_total_rate(self):
        
        translations = self.find_translations()

        total_rate = 0
        for option in translations:
            if option == "Rotation":
                diffusion_rate = 0.1
                step_size = self.rot_step_size
            elif option == "Translation_X":
                diffusion_rate = self.rate_translation_x
                step_size = self.trans_step_size
                
            elif option == "Translation_Y":
                diffusion_rate = self.rate_translation_y
                step_size = self.trans_step_size

            total_rate += self.calculate_rate(step_size, diffusion_rate, option)
            print("Total", total_rate)
        return total_rate

def random_number_generator():
    random_num = np.random.uniform(0, 1)
    return random_num

def move_dbtf(dbtf, grid_size=16):
    transition = random_number_generator()
    direction = random_number_generator()

    # Probabilistic choice between x and y movement
    if transition < dbtf.rate_translation_x / dbtf.total_rate:
        print("X movement")
        step = dbtf.trans_step_size if direction >= 0.5 else -dbtf.trans_step_size

        # Rotate x movement based on angle
        dx = step * np.cos(np.radians(dbtf.angle))
        dy = step * np.sin(np.radians(dbtf.angle))

    elif transition < (dbtf.rate_translation_x + dbtf.rate_translation_y) / dbtf.total_rate:
        print("Y movement")
        step = dbtf.trans_step_size if direction >= 0.5 else -dbtf.trans_step_size

        # Rotate y movement based on angle (perpendicular to x)
        dx = -step * np.sin(np.radians(dbtf.angle))
        dy = step * np.cos(np.radians(dbtf.angle))

    else:
        print("Rotation")
        if direction < 0.5:
            dbtf.angle -= dbtf.rot_step_size
        else:
            dbtf.angle += dbtf.rot_step_size
        return  # No translation if rotation occurs

    # Update position
    dbtf.position[0] += dx
    dbtf.position[1] += dy

    # Keep the box within bounds
    if dbtf.position[0] > grid_size or dbtf.position[0] < -grid_size:
        dbtf.position[0] = -dbtf.position[0]
    if dbtf.position[1] > grid_size or dbtf.position[1] < -grid_size:
        dbtf.position[1] = -dbtf.position[1]

    return 0

def generate_hexagonal_lattice(grid_size, a=1):
    """
    Generate a hexagonal lattice of points within a given grid size.
    
    Args:
        grid_size: Number of rows and columns to generate.
        a: Spacing between adjacent points.
    
    Returns:
        A list of (x, y) tuples representing the hexagonal lattice points.
    """
    lattice = []
    for i in range(-grid_size, grid_size + 1):
        for j in range(-grid_size, grid_size + 1):
            x = i * a
            y = j * a * np.sqrt(3) / 2
            if j % 2 != 0:  # Offset for staggered rows
                x += a / 2
            lattice.append((x, y))
    return lattice



def main():
    grid_length = 16
    time = 0
    time_limit = 100
    dbtf = DBTF_1D()

    positions_x = [dbtf.position[0]]
    positions_y = [dbtf.position[1]]
    times = [0]
    colourbar_added = False

    def update(frame):
        nonlocal time, colourbar_added

        if time >= time_limit:
            ani.event_source.stop()  # Stop the animation when the time limit is reached
            return

        total_rate = dbtf.total_rate

        # Calculate delta_time
        delta_time = -(1 / total_rate) * np.log(random_number_generator())
        time += delta_time

        # Move the molecule
        move_dbtf(dbtf)

        # Append positions
        positions_x.append(dbtf.position[0])
        positions_y.append(dbtf.position[1])
        times.append(time)

        # Clear the plot and redraw
        
        ax.clear()
        ax.set_xlim(-grid_length, grid_length)
        ax.set_ylim(-grid_length, grid_length)
        ax.set_title(f"Time: {time:.2f}")

        # Draw the path with a color gradient
        sc = ax.scatter(
            positions_x,
            positions_y,
            c=times,  # Use time as the color
            cmap="viridis",  # Color map for the gradient
            edgecolor="k",
            s=50  # Marker size
        )



        # Add a colorbar to indicate time
        if not colourbar_added:
            plt.colorbar(sc, ax=ax, label="Time")
            colourbar_added = True

        # Add the rectangle
        rect = Rectangle(
            (dbtf.position[0]-0.5, dbtf.position[1]-0.25),
            1,  # Width
            0.5,  # Height
            edgecolor="red",
            facecolor="orange",
            alpha=0.8,
        )
        transform = (
            Affine2D()
            .rotate_deg_around(dbtf.position[0], dbtf.position[1], dbtf.angle) + ax.transData
        )
        rect.set_transform(transform)

        ax.add_patch(rect)

    # Set up the plot
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=np.arange(1000000), repeat=False)
    plt.show()

    
    return 0

if __name__ == "__main__":
   main()