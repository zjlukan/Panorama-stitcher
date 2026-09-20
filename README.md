# Features  
**Automatic sequencing:**  
* Computes most likely image pairings for each image, then uses a probabilistic model to determine whether or not to keep the pairing  
* Images are paired independent of the sequence they were uploaded in  

**Bundle Adjustment:**
* Simply concatenating pairwise homographies leads to issues such as accumulated error  
* Bundle adjustment is used instead to solve for camera parameters jointly, where the sum squared reprojection error is minimized using Levenberg-Marquardt  

**Blending**
* Uses a graph-cut seam optimizer to find the least intrusive line of stitching
* Images are blended using multi-band blending to minimize the appearance of edges

**Front end**
* Streamlit is used for the browser interface

## Dependencies: 
**Python3**
**numpy**
**ctypes**

## Running the code:
To run the code, run the following command:

python .\pano_stitcher.py [number of images] [image 1 path] ... [image n path] [show steps]

example:

python .\pano_stitcher.py 3 image1.png image2.png image3.png 1

**Show steps:** either 0 (false) or 1 (true), displays all of the steps in the panorama stitching process
