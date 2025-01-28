import matplotlib.pyplot as plt
import numpy as np
from nexusformat.nexus import nxload
import imageio

number = 7050
path = rf"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\CrCl3\mm38550-2\i06-2-{number}.nxs"

nexus = nxload(path)
print(nexus.tree)

images = nexus['/entry/medipix/data']

# plt.imshow(iraise NeXusError("Use slabs to access data larger than "
# nexusformat.nexus.tree.NeXusError: Use slabs to access data larger than NX_MEMORY=2000 MBmages[2999], cmap='gray')
# plt.show()

slab_size = 100  # Adjust this size based on your memory capacity

# Process images in slabs
num_images = images.shape[0]
for start in range(0, 2*slab_size, slab_size):
    end = min(start + slab_size, num_images)
    slab = images[start:end]
    print(f'Processing images {start} to {end}')
    for i, image in enumerate(slab):
        image_index = start + i

        # normalized_image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
        normalized_image = image.astype(np.uint8)


        imageio.imwrite(rf'C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\CrCl3\HOPG\7050\image_{image_index:04d}.png', normalized_image)
        if i == 99:
            plt.imshow(image, cmap='gray')
            plt.show()
