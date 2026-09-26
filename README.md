# Description
Image stitching is the process of combining multiple images with overlapping fields to produce a composite image that has a wider field of view than a single image. It is used in a variety of fields, such as satellite imaging and medical imaging. This project provides an easy way of stitching a sequence of ordered images and seamlessly blending them. OpenCV is utilized in this project, but none of the functions from the stitcher class were used.

# Features  
**Homography Fitting:**  
* Features matched using SIFT are run through a ratio test to eliminate ambiguous matches  
* Homographies are estimated using RANSAC algorithm for invariance to outliers  

**Image Warping + Mosiacing:**
* Uses a running panorama and iteratively builds it with each image in order

**Blending**
* Laplacian pyramid blending merges images frequency-band by frequency-band, eliminating the hard seams a flat crossfade would leave behind

**Front end**
* Streamlit is used for the browser interface. Images are uploaded and assigned an order and the final image can be downloaded as a PNG.

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
