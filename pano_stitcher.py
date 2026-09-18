# Given a list of images (in any order), stitches them together using feature matching to create a panorama


import cv2
import numpy as np
import numpy.linalg
import tkinter as tk
import sys
import blending


def extract_features(images):
    """
    Uses SIFT to find all the features for each image
    :param images: a list containing 4-channel images
    :return: a list features where features[i] = list of [keypoints, descriptor] pairs for image i
    """

    # use SIFT to get key points and descriptors for the image
    # lower edgeThreshold: less tolerance for edges
    # higher contrastThreshold: less tolerance for areas of low contrast
    sift = cv2.SIFT_create(edgeThreshold=20, contrastThreshold=0.01)

    features = []
    for x in range(0, len(images)):
        features.append(sift.detectAndCompute(images[x], None))

    return features


def image_match(idx, images, features):
    """
    Find the for the given image (images[idx]), find the n images with the most feature matches (n=6 by default)
    :param features: a list features where features[i] = [list of keypoints, list of descriptors] pairs for image i
    :param idx: the index of the image to be matched to the other images in the list
    :param images: list of the other 4-channel images which will be matched to images[idx]
    :return: a list of indexes corresponding to the images with the most matches
    """
    POTENTIAL_MATCHES = 6

    img_des = []  # contains the feature descriptors of image
    des = []  # contains the feature descriptors of all other images
    best_imgs = []  # the number of feature matches per image
    des_idx = []  # des_idx[i] corresponds to the index of the image of des[i]

    for x in range(0, len(features)):
        if x == idx:
            img_des = features[x][1]
            best_imgs.append([x, -1])
        else:
            des.extend(features[x][1])
            des_idx.extend([x]*len(features[x][1]))
            best_imgs.append([x, 0])

    des = np.array(des)

    # find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(img_des, des, k=4)

    # for each match, find which image it corresponds to and iterate best_imgs by 1
    for m in matches:
        for x in range(4):
            best_imgs[des_idx[m[x].trainIdx]][1] += 1

    # sort the images by the number of features they have in common with image, take the 6 best ones
    best_imgs = sorted(best_imgs, key=lambda x: x[1])

    # don't bother with finding the best matches if there are fewer images than the threshold
    if len(images) <= POTENTIAL_MATCHES + 1:
        return best_imgs[1:]

    best_imgs = best_imgs[-6:]
    return best_imgs


def match_verification(idx, images, features, best_imgs):
    """
    Use a probabilistic model to determine using the number of inliers and the total number of features
    between images whether to pair them or not
    :param idx:
    :param images:
    :param features:
    :param best_imgs:
    :return:
    """
    a = 8
    b = 0.3
    match = []
    kpoints = features[idx][0]

    for l in best_imgs:
        i = l[0]
        kpoints2 = features[i][0]
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(features[idx][1], features[i][1], k=2)
        goodm = []

        # use only the matches that have sufficient distance between best and 2nd best match to reduce ambiguous matches
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                goodm.append([m])

        # get keypoints
        pts1 = np.float32([kpoints[m[0].queryIdx].pt for m in goodm])
        pts2 = np.float32([kpoints2[m[0].trainIdx].pt for m in goodm])

        # findHomography returns a matrix and a mask, where a 1 denotes an inlier and a 0 denotes an outlier
        h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

        n_inliers = sum(mask)
        n_features = len(mask)

        if n_inliers > a + b*n_features:
            match.append(True)
        else:
            match.append(False)
    return match


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
        result = blending.blend_images(result, img1, 12)

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
        # h = homography(img2_copy, img_copy, show)
        # warp = warp_image(img2, h[0], show)
        # img = create_mosaic([img, warp[0]], [[0, 0], warp[1]], show)

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


def process_images(img_paths):
    images = []
    for i in img_paths:
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        images.append(img)
    f = extract_features(images)
    image_match(0, images, f)

imgs = []
for x in range(0,10):
    imgs.append("test_data_1/medium0" + str(x) + ".jpg")
for x in range(10, 13):
    imgs.append("test_data_1/medium" + str(x) + ".jpg")
process_images(imgs)


'''
i = []
inp = ""
n = 0
show_steps = True
if len(sys.argv) < 3:
    raise ValueError("not enough arguments")
try:
    n = int(sys.argv[1])
except ValueError:
    print("n is not a valid number")
if len(sys.argv) > n + 2:
    if sys.argv[n + 2] != "0":
        show_steps = True
for x in range(2, n + 2):
    i.append(sys.argv[x])

image_final = create_pano(i, show_steps)
cv2.imshow("image", image_final)
cv2.waitKey(0)
'''
