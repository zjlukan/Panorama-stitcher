"""
Streamlit front end for the ordered-pair panorama stitcher in panorama_core.py.

Images are stitched one at a time, in the order you specify, by folding each new image
onto the running panorama with homography() -> warp_image() -> create_mosaic()
(Laplacian-pyramid multi-band blend), exactly as panorama_core.create_pano() does --
this app just replaces file paths / cv2.imshow / tkinter with a web UI.
"""

from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from pano_core import homography, warp_image, create_mosaic


st.set_page_config(page_title="Panorama Stitcher", page_icon="🖼️", layout="wide")

st.title("🖼️ Panorama Stitcher")
st.caption("SIFT feature matching + homography warping + multi-band Laplacian blending")

with st.sidebar:
    st.header("Options")
    show_matches = st.checkbox("Show feature matches for each pair", value=False)
    st.caption(
        "Stitching folds each image, in order, onto the running panorama. "
        "Best results come from images with real overlap, taken from roughly the same viewpoint."
    )


def load_bgra(uploaded_file) -> np.ndarray:
    uploaded_file.seek(0)
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode {uploaded_file.name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)


def bgra_to_rgba(img_bgra: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)


def draw_matches(img_a_bgr: np.ndarray, img_b_bgr: np.ndarray) -> np.ndarray:
    """Visualization only -- recomputes SIFT + knn matches to draw them; does not feed the stitch."""
    sift = cv2.SIFT_create(edgeThreshold=20, contrastThreshold=0.01)
    kp1, des1 = sift.detectAndCompute(img_a_bgr, None)
    kp2, des2 = sift.detectAndCompute(img_b_bgr, None)
    bf = cv2.BFMatcher()
    knn = bf.knnMatch(des1, des2, k=2)
    good = [[m] for m, n in knn if m.distance < 0.5 * n.distance]
    vis = cv2.drawMatchesKnn(
        img_a_bgr, kp1, img_b_bgr, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


uploaded_files = st.file_uploader(
    "Upload the images to stitch",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload 2 or more overlapping images to get started.")
    st.stop()

if len(uploaded_files) < 2:
    st.warning("Add at least one more image -- stitching needs 2 or more.")
    st.stop()

st.subheader("1. Set the stitching order")
st.caption("Give each image a position. Ties are broken by upload order.")

order_map = {}
thumb_cols = st.columns(min(len(uploaded_files), 4))
for idx, f in enumerate(uploaded_files):
    col = thumb_cols[idx % len(thumb_cols)]
    f.seek(0)
    col.image(f, caption=f.name, use_container_width=True)
    order_map[f.name] = col.number_input(
        "Position",
        min_value=1,
        max_value=len(uploaded_files),
        value=idx + 1,
        step=1,
        key=f"order_{idx}_{f.name}",
    )

ordered_names = sorted(order_map, key=lambda n: order_map[n])
file_by_name = {f.name: f for f in uploaded_files}
ordered_files = [file_by_name[n] for n in ordered_names]

st.subheader("2. Stitch")
if st.button("Stitch panorama", type="primary"):
    progress = st.progress(0.0, text="Loading images...")
    try:
        images = [load_bgra(f) for f in ordered_files]
    except ValueError as e:
        progress.empty()
        st.error(str(e))
        st.stop()

    pano = images[0]
    n_pairs = len(images) - 1

    for i in range(n_pairs):
        progress.progress(i / n_pairs, text=f"Matching image {i + 2} of {len(images)}...")

        next_bgra = images[i + 1]
        next_bgr = cv2.cvtColor(next_bgra, cv2.COLOR_BGRA2BGR)
        pano_bgr = cv2.cvtColor(pano, cv2.COLOR_BGRA2BGR)

        if show_matches:
            with st.expander(f"Feature matches: image {i + 2} vs. running panorama"):
                st.image(draw_matches(next_bgr, pano_bgr), use_container_width=True)

        h = homography(next_bgr, pano_bgr, show=False)
        warp, corner = warp_image(next_bgra, h[0], show=False)
        pano = create_mosaic([pano, warp], [[0, 0], corner], show=False)

        progress.progress((i + 1) / n_pairs, text=f"Blended image {i + 2} of {len(images)}")

    progress.empty()
    st.success("Panorama complete!")
    st.image(bgra_to_rgba(pano), caption="Final panorama", use_container_width=True)

    pil_img = Image.fromarray(bgra_to_rgba(pano))
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    st.download_button(
        "Download panorama (PNG)",
        data=buf.getvalue(),
        file_name="panorama.png",
        mime="image/png",
    )