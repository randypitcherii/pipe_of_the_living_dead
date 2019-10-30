```python
import os

import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.feature import blob_dog, blob_doh
from skimage.io import imread
from skimage.util import invert
```


```python
def get_preprocessed_img(img_path):
    """
    This function accepts a path to an image
    file and returns a tuple containing the 
    unaltered rgb image and the preprocessed image
    """
    # read image 
    img_rgb = imread(img_path)
    
    # make image gray
    img_gray = rgb2gray(img_rgb)
    
    # Contrast stretching
    p_start, p_end = np.percentile(img_gray, (1, 15))
    img_gray_rescale = rescale_intensity(img_gray, in_range=(p_start, p_end))
    
    # invert image
    img_gray_rescale_inverted = invert(img_gray_rescale)
    
    return (img_rgb, img_gray_rescale_inverted)
    
    
def doh_detect_pipes(img_path, show=False):
    """
    This function accepts a path to an image
    file and returns the int number of pipes
    detected in the image.
    
    If the optional show param is set to true, the resuls
    are plotted visually before returning pipe counts.
    """
    # get images
    img_rgb, img_preprocessed = get_preprocessed_img(img_path)
    
    # detect pipes
    pipes = blob_doh(img_preprocessed, max_sigma=150, 
                         min_sigma=45, threshold=0.005, overlap=0.01)
    
    # plot detection
    if show:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img_rgb)
        ax.set_axis_off()
        
        for pipe in pipes:
            y, x, r = pipe
            c = plt.Circle((x, y), r, color='cyan', linewidth=2, fill=False)
            ax.add_patch(c)
            
        plt.show()
    
    return len(pipes)
```


```python
# get all image files
imgs_dir = '../images/'
img_and_video_files = os.listdir(imgs_dir)
img_files = filter(lambda filename: filename.endswith('.jpg') or filename.endswith('.JPG'), img_and_video_files)

# find pipes
for img_file in img_files:
    path = imgs_dir + img_file
    num_pipes = doh_detect_pipes(path, show=True)
    print(f"Found {num_pipes} in {img_file}\n\n")
```


![png](output_2_0.png)


    Found 13 in IMG_0606.JPG
    
    



![png](output_2_2.png)


    Found 18 in IMG_0612.JPG
    
    



![png](output_2_4.png)


    Found 17 in IMG_0613.JPG
    
    



![png](output_2_6.png)


    Found 20 in IMG_0607.JPG
    
    



![png](output_2_8.png)


    Found 17 in IMG_0834.JPG
    
    



![png](output_2_10.png)


    Found 16 in IMG_0598.JPG
    
    



![png](output_2_12.png)


    Found 18 in IMG_0611.JPG
    
    



![png](output_2_14.png)


    Found 17 in IMG_0605.JPG
    
    



![png](output_2_16.png)


    Found 11 in IMG_0639.JPG
    
    



![png](output_2_18.png)


    Found 12 in IMG_0638.JPG
    
    



![png](output_2_20.png)


    Found 21 in IMG_0604.JPG
    
    



![png](output_2_22.png)


    Found 13 in IMG_0610.JPG
    
    



![png](output_2_24.png)


    Found 17 in IMG_0599.JPG
    
    



![png](output_2_26.png)


    Found 21 in IMG_0589.JPG
    
    



![png](output_2_28.png)


    Found 18 in IMG_0833.JPG
    
    



![png](output_2_30.png)


    Found 16 in IMG_0628.JPG
    
    



![png](output_2_32.png)


    Found 21 in IMG_0614.JPG
    
    



![png](output_2_34.png)


    Found 19 in IMG_0600.JPG
    
    



![png](output_2_36.png)


    Found 21 in IMG_0601.JPG
    
    



![png](output_2_38.png)


    Found 15 in IMG_0615.JPG
    
    



![png](output_2_40.png)


    Found 15 in IMG_0629.JPG
    
    



![png](output_2_42.png)


    Found 20 in IMG_0588.JPG
    
    



![png](output_2_44.png)


    Found 19 in IMG_0603.JPG
    
    



![png](output_2_46.png)


    Found 18 in IMG_0617.JPG
    
    



![png](output_2_48.png)


    Found 18 in IMG_0616.JPG
    
    



![png](output_2_50.png)


    Found 19 in IMG_0602.JPG
    
    



![png](output_2_52.png)


    Found 7 in sample_cropped.jpg
    
    



![png](output_2_54.png)


    Found 27 in IMG_0659.JPG
    
    



![png](output_2_56.png)


    Found 26 in IMG_0658.JPG
    
    



![png](output_2_58.png)


    Found 41 in IMG_0664.JPG
    
    



![png](output_2_60.png)


    Found 5 in IMG_5654.JPG
    
    



![png](output_2_62.png)


    Found 33 in IMG_0663.JPG
    
    



![png](output_2_64.png)


    Found 33 in IMG_0662.JPG
    
    



![png](output_2_66.png)


    Found 42 in IMG_0648.JPG
    
    



![png](output_2_68.png)


    Found 37 in IMG_0660.JPG
    
    



![png](output_2_70.png)


    Found 29 in IMG_0661.JPG
    
    



![png](output_2_72.png)


    Found 38 in IMG_0649.JPG
    
    



![png](output_2_74.png)


    Found 14 in IMG_0644.JPG
    
    



![png](output_2_76.png)


    Found 35 in IMG_0650.JPG
    
    



![png](output_2_78.png)


    Found 34 in IMG_0651.JPG
    
    



![png](output_2_80.png)


    Found 38 in IMG_0653.JPG
    
    

