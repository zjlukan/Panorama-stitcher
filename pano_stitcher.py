# Given a list of images (in any order), stitches them together using feature matching to create a panorama


import cv2
import numpy as np
import numpy.linalg
import tkinter as tk
import sys


def create_gauss(img, levels):
    """
    create a Gaussian (resolution) pyramid
    :param img: a copy of the original image that will become the first layer of the pyramid
    :param levels: the number of levels in the pyramid
    :return: a Gaussian pyramid represented by list of images with length levels
    """
    gauss = []
    layer = img.copy()
    for x in range(levels):
        gauss.append(layer)
        layer = cv2.pyrDown(layer)
    return gauss


def create_laplacian(gauss, levels):
    """
    create a Laplacian (scaled image details) pyramid from a Gaussian pyramid
    :param gauss: a Gaussian pyramid represented by list of images with length levels
    :param levels: the number of levels in the pyramid
    :return: a Laplacian pyramid represented by list of images with length levels
    """
    laplacian = []
    for x in range(levels - 1):
        up = cv2.pyrUp(gauss[x + 1])
        up = cv2.resize(up, (gauss[x].shape[1], gauss[x].shape[0]))
        resid = cv2.subtract(gauss[x], up)
        laplacian.append(resid)
    laplacian.append(gauss[-1])
    return laplacian


def create_mask(image):
    """
    create a distance-based mask from image
    :param image: 4-channel image
    :return: mask with same height and width as input image with values between 0 and 1 representing the distance from
    the nearest pixel with non-zero alpha
    """
    # create binary mask based on alpha channel of image
    mask = (image[:, :, 3] > 1).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    # get rid of holes
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return dist


def homography(image_a, image_b, show=False):
    """
    Uses SIFT and k nearest neighbors to compute a homography between 2 images
    :param show: display steps
    :param image_a: first 4-channel image
    :param image_b: second 4-channel image
    :return: homography matrix
    """
    # use SIFT to get key points and descriptors for the 2 images
    # lower edgeThreshold: less tolerance for edges, higher contrastThreshold: less tolerance for areas of low contrast
    sift = cv2.SIFT_create(edgeThreshold=20, contrastThreshold=0.01)
    kpoints, des = sift.detectAndCompute(image_a, None)
    kpoints2, des2 = sift.detectAndCompute(image_b, None)

    # find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des, des2, k=2)

    # use only the matches that have sufficient distance between best and second-best match to reduce ambiguous matches
    goodm = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            goodm.append([m])

    # use findHomography to get the 3x3 matrix
    pts1 = np.float32([kpoints[m[0].queryIdx].pt for m in goodm])
    pts2 = np.float32([kpoints2[m[0].trainIdx].pt for m in goodm])
    h = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    if show:
        # display keypoints
        kpoints_a_i = cv2.drawKeypoints(image_a, kpoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kpoints_b_i = cv2.drawKeypoints(image_b, kpoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # display matches
        matches_i = cv2.drawMatchesKnn(image_a, kpoints, image_b, kpoints2, matches, None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # display good matches
        goodm_i = cv2.drawMatchesKnn(image_a, kpoints, image_b, kpoints2, goodm, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # show images
        cv2.imshow("Image_a keypoints", kpoints_a_i)
        cv2.waitKey(0)
        cv2.imshow("Image_b keypoints", kpoints_b_i)
        cv2.waitKey(0)
        cv2.imshow("Knn matched keypoints", matches_i)
        cv2.waitKey(0)
        cv2.imshow("Good matches", goodm_i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return h


def warp_image(image, homography, show=False):
    """
    Use the computed homography to warp the image, while keeping it properly cropped
    :param show: display steps
    :param image: the first 4-channel image passed to homography()
    :param homography: 3x3 matrix
    :return: the warped 4-channel image minimally cropped, and the coordinates of the upper left corner of the warped
        image in the general plane
    """
    # corners: upper left, upper right, lower right, lower left
    corners = [np.array([[[0, 0]]], dtype='float32'),
               np.array([[[image.shape[1] - 1, 0]]], dtype='float32'),
               np.array([[[image.shape[1] - 1, image.shape[0] - 1]]], dtype='float32'),
               np.array([[[0, image.shape[0] - 1]]], dtype='float32')]

    # find the bounds of the transformed image by transforming the corners with homography
    for c in range(0, len(corners)):
        corners[c] = cv2.perspectiveTransform(corners[c], homography)
        corners[c] = corners[c][0][0]
    minX = corners[0][0]
    maxX = corners[0][0]
    minY = corners[0][1]
    maxY = corners[0][1]

    # find where the smallest and largest x and y coordinate pixels will be
    for c in corners:
        minX = int(min(c[0], minX))
        minY = int(min(c[1], minY))
        maxX = int(max(c[0], maxX))
        maxY = int(max(c[1], maxY))

    # proportions of image after homography
    w = maxX - minX
    h = maxY - minY

    # create a translation matrix so that the image stays in bounds
    xShift = -minX
    yShift = -minY

    T = np.float64([
        [1, 0, xShift],
        [0, 1, yShift],
        [0, 0, 1]
    ])

    # image is warped by homography, then translated so that it fits within the correct bounds
    M = T @ homography
    warp = cv2.warpPerspective(image, M, (w, h))

    # pixels not covered by warped image are turned transparent
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2BGRA)

    for m in range(0, warp.shape[0]):
        for n in range(0, warp.shape[1]):
            if warp[m, n, 0] < 1:
                warp[m, n, 3] = 0

    invT = np.linalg.inv(T)
    ULcorner = np.array([[[0, 0]]], dtype='float32')
    ULcorner = cv2.perspectiveTransform(ULcorner, invT)

    # return the warped image and the coordinates of the top left corner
    if show:
        # display images
        cv2.imshow("Image before warp", image)
        cv2.waitKey(0)
        cv2.imshow("Image after warp", warp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return [warp, ULcorner[0][0]]


def blend_images(image_a, image_b, levels, show=False):
    """
    Uses Laplacian pyramids to blend 2 images together
    :param show: display steps
    :param image_a: 4-channel image, will correspond to 1 in the mask
    :param image_b: 4-channel image, will correspond to 0 in the mask
    :param levels: the number of levels in the Laplacian pyramid
    :return: 4-channel image that blends a and b
    """
    # each pixel in the mask has value in [0,1] corresponding to the distance from the image
    distA = create_mask(image_a)
    distB = create_mask(image_b)
    normA = cv2.normalize(distA, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    normB = cv2.normalize(distB, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # combine both masks
    mask = distA / (distA + distB + 1e-8)

    # process the rgb part of the image separately
    a_rgb = image_a[:, :, :3].astype(np.float32) / 255.0
    b_rgb = image_b[:, :, :3].astype(np.float32) / 255.0

    # create gaussian and laplacian pyramids for both images and the mask
    m_gauss = create_gauss(mask, levels)
    a_gauss = create_gauss(a_rgb, levels)
    b_gauss = create_gauss(b_rgb, levels)

    a_laplacian = create_laplacian(a_gauss, levels)
    b_laplacian = create_laplacian(b_gauss, levels)

    # blend all levels of each laplacian according to the weights in the mask
    blend = []
    for x in range(levels):
        blend.append(a_laplacian[x] * m_gauss[x][..., np.newaxis] +
                     b_laplacian[x] * (1 - m_gauss[x][..., np.newaxis]))

    # collapse the combined pyramid
    img = blend[-1]
    for x in range(len(blend) - 2, -1, -1):
        up_i = cv2.pyrUp(img)
        up_i = cv2.resize(up_i, (blend[x].shape[1], blend[x].shape[0]))
        img = up_i + blend[x]

    # convert image to uint8 4-channel with black regions set to alpha 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask2 = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    img[mask2, 3] = 0
    img = cv2.convertScaleAbs(img, alpha=255)

    if show:
        # display images
        cv2.imshow("distA", normA)
        cv2.waitKey(0)
        cv2.imshow("distB", normB)
        cv2.waitKey(0)
        cv2.imshow("final image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


def create_mosaic(images, origins, show=False):
    """
    Given a list of 4-channel images and a list of coordinates corresponding to their upper left corners, combine them
        onto a common plane and blend them together
    :param show: display steps
    :param images: list of 4-channel images
    :param origins: list of [x,y] coordinates of the images in the common plane
    :return: one 4-channel image containing all the images blended together
    """
    # find the size of the final image
    img = images[0].copy()
    minX = origins[0][0]
    minY = origins[0][1]
    maxX = origins[0][0]
    maxY = origins[0][1]
    for i in range(0, len(images)):
        minX = min(minX, origins[i][0])
        minY = min(minY, origins[i][1])
        maxX = max(maxX, origins[i][0] + images[i].shape[1])
        maxY = max(maxY, origins[i][1] + images[i].shape[0])

    # create an empty image with the required dimensions
    w = int(maxX - minX)
    h = int(maxY - minY)
    result = np.zeros((h, w, 4), dtype=np.uint8)

    # place the first image
    x = int(origins[0][0] - minX)
    y = int(origins[0][1] - minY)
    result[y:y + images[0].shape[0], x:x + images[0].shape[1]] = images[0]

    # replace pixels in the empty image with the non-transparent ones in each image
    for i in range(1, len(images)):
        img1 = np.zeros((h, w, 4), dtype=np.uint8)
        x = int(origins[i][0] - minX)
        y = int(origins[i][1] - minY)
        img1[y:y + images[i].shape[0], x:x + images[i].shape[1]] = images[i]
        result = blend_images(result, img1, 12)

    if show:
        cv2.imshow("Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result


def create_pano(images, show=False):
    """
    Given a list of image paths, build a panorama
    :param show: show the final image
    :param images: list of strings of the image paths
    :return: 4-channel panorama image
    """
    img = cv2.imread(images[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    for x in range(1, len(images)):

        # iteratively stitch together each image in images
        img2 = cv2.imread(images[x])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
        img2_copy = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
        img_copy = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h = homography(img2_copy, img_copy, show)
        warp = warp_image(img2, h[0], show)
        img = create_mosaic([img, warp[0]], [[0, 0], warp[1]], show)

    # adjust the image to proper size to fit screen
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    ratio = min(width / img.shape[0], height / img.shape[1])
    w = int(img.shape[1] * ratio)
    h = int(img.shape[0] * ratio)
    img = cv2.resize(img, (w, h))

    if show:
        cv2.imshow("final image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


i = []
inp = ""
n = 0
show_steps = False
if len(sys.argv) < 3:
    raise ValueError("not enough arguments")
try:
    n = int(sys.argv[1])
except ValueError:
    print("n is not a valid number")
if len(sys.argv) > n+2:
    if sys.argv[n+2] != "0":
        show_steps = True
for x in range(2, n+2):
    i.append(sys.argv[x])

image_final = create_pano(i, show_steps)
cv2.imshow("image", image_final)
cv2.waitKey(0)
