import cv2
import numpy as np


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