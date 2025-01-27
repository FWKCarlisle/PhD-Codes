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

for i, image in enumerate(images):
    imageio.imwrite(rf'C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\CrCl3\HOPG\7050\image_{i:04d}.png', image)

