from PIL import Image, ImageChops

img = Image.open('test.jpg')
 
# Passing the image object to invert()  
inv_img = ImageChops.invert(img)
 
# Displaying the output image
inv_img.show()