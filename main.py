import argparse
import cv2
import pano_core

parser = argparse.ArgumentParser()
parser.add_argument("--show", default=False, action='store_true')
parser.add_argument("image_paths", type=str, nargs='*')

args = parser.parse_args()

if len(args.image_paths) < 2:
    raise ValueError("Stitcher requires at least 2 photos")

img = pano_core.create_pano(args.image_paths, args.show)

cv2.imshow("Final pano", img)
cv2.waitKey(0)
