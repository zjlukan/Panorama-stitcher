"""Jointly refine the homographies of ALL images into one reference frame.

Pipeline:
    pairs = match_all_pairs(images, features)      # verified pairwise matches + RANSAC inliers
    H_init = initial_homographies(len(images), pairs, ref=0)   # chain pairwise H's outward from the reference
    H_all = refine_homographies(pairs, H_init, ref=0)          # least-squares over every pair at once

H_all[k] maps image k -> reference image's frame (same convention as your warp_image()).
`features` is what extract_features() returns: features[k] = (keypoints, descriptors).
"""
from collections import deque

import cv2
import numpy as np
from scipy.optimize import least_squares

RATIO = 0.8   # Lowe ratio test threshold
ALPHA = 8.0   # match verification: n_i > ALPHA + BETA * n_f
BETA = 0.3


# ---------- pairwise matching ----------

def in_img(pts, H, w, h):
    """Boolean mask: which points does H map inside a w x h image?"""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = p[:, :2] / z[:, None]
    return (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)


def count_overlap(H, pts1, pts2, img1, img2):
    """n_f: number of matches whose two points both lie in the overlap. H maps img1 -> img2."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    in1 = in_img(pts1, H, w2, h2)
    in2 = in_img(pts2, np.linalg.inv(H), w1, h1)
    return int((in1 & in2).sum())


def match_pair(feat_i, feat_j, img_i, img_j):
    """Ratio-test matches, RANSAC homography, and the paper's verification.

    Returns (H_ij, inlier_pts_i, inlier_pts_j) with H_ij mapping image i -> image j,
    or None if the pair is not a verified match.
    """
    kp_i, des_i = feat_i
    kp_j, des_j = feat_j

    knn = cv2.BFMatcher().knnMatch(des_i, des_j, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < 4:
        return None

    pts_i = np.float32([kp_i[m.queryIdx].pt for m in good])
    pts_j = np.float32([kp_j[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC)
    if H is None:
        return None

    n_inliers = int(mask.sum())
    n_features = count_overlap(H, pts_i, pts_j, img_i, img_j)
    if n_inliers <= ALPHA + BETA * n_features:
        return None

    inliers = mask.ravel().astype(bool)
    return H, pts_i[inliers], pts_j[inliers]


def match_all_pairs(images, features):
    """Verified matches for every image pair (i < j). Returns {(i, j): (H_ij, pts_i, pts_j)}."""
    pairs = {}
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            result = match_pair(features[i], features[j], images[i], images[j])
            if result is not None:
                pairs[(i, j)] = result
    return pairs


# ---------- initialization ----------

def initial_homographies(n_images, pairs, ref=0):
    """Chain pairwise homographies outward from the reference image (breadth-first search).

    Returns {image_index: H} for every image connected to the reference; H maps that image
    into the reference frame. Images with no verified path to the reference are left out.
    """
    # neighbors[a] holds (b, M) where M maps image b -> image a
    neighbors = {k: [] for k in range(n_images)}
    for (i, j), (H_ij, _, _) in pairs.items():
        neighbors[j].append((i, H_ij))
        neighbors[i].append((j, np.linalg.inv(H_ij)))

    H = {ref: np.eye(3)}
    queue = deque([ref])
    while queue:
        cur = queue.popleft()
        for nxt, M_nxt_to_cur in neighbors[cur]:
            if nxt not in H:
                H[nxt] = H[cur] @ M_nxt_to_cur
                H[nxt] /= H[nxt][2, 2]
                queue.append(nxt)
    return H


# ---------- refinement ----------

def pack(H, order):
    """Homographies -> flat vector of 8 free parameters per image (H[2,2] fixed to 1)."""
    return np.concatenate([(H[k] / H[k][2, 2]).ravel()[:8] for k in order])


def unpack(x, order, ref):
    """Inverse of pack(). The reference image is fixed to the identity."""
    H = {ref: np.eye(3)}
    for n, k in enumerate(order):
        H[k] = np.append(x[8 * n: 8 * n + 8], 1.0).reshape(3, 3)
    return H


def residuals(x, pairs, order, ref):
    """Reprojection error in pixels: for each pair, map image i's inliers into image j and compare."""
    H = unpack(x, order, ref)
    errors = []
    for (i, j), (_, pts_i, pts_j) in pairs.items():
        H_ij = np.linalg.inv(H[j]) @ H[i]   # image i -> reference -> image j
        proj = cv2.perspectiveTransform(pts_i.reshape(-1, 1, 2).astype(np.float64), H_ij)
        errors.append((proj.reshape(-1, 2) - pts_j).ravel())
    return np.concatenate(errors)


def refine_homographies(pairs, H_init, ref=0, huber_px=2.0):
    """Jointly optimize all homographies. Returns {image_index: H} mapping each image -> reference."""
    order = [k for k in sorted(H_init) if k != ref]
    used = {ij: v for ij, v in pairs.items() if ij[0] in H_init and ij[1] in H_init}

    result = least_squares(
        residuals,
        pack(H_init, order),
        args=(used, order, ref),
        method="trf",
        loss="huber",
        f_scale=huber_px,   # residuals above this many pixels are down-weighted
        x_scale="jac",      # translation and perspective terms differ by orders of magnitude
    )
    return unpack(result.x, order, ref)