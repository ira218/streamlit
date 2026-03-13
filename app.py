import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Gum + Plaque + Cavity Detector", layout="centered")
st.title("Image Analysis")

uploaded = st.file_uploader("Upload a tooth image", type=["jpg", "jpeg", "png"])

tabs = st.tabs(["Plaque", "Cavity", "Gums"])

# -----------------------------
# Hard-coded thresholds
# -----------------------------
PLAQUE_H_LOW   = 10
PLAQUE_H_HIGH  = 40
PLAQUE_S_MIN   = 60
PLAQUE_B_MIN   = 140
PLAQUE_L_MAX   = 220

BLACK_V_MAX     = 100
BLACK_S_MAX     = 180
CONTRAST_THRESH = 25
MIN_BLACK_AREA  = 50
OVERLAP_RATIO   = 0.80
RING_RATIO      = 0.65
MIN_DIST        = 3

INFLAMED_A_MIN    = 176
INFLAMED_S_MIN    = 70
MIN_INFLAMED_AREA = 150

with tabs[0]:
    show_plaque_overlay = st.checkbox("Show plaque overlay", value=True)
with tabs[1]:
    show_black_overlay = st.checkbox("Show black spot overlay", value=True)
with tabs[2]:
    show_inflamed_overlay = st.checkbox("Show inflamed overlay", value=True)


# -----------------------------
# Helpers
# -----------------------------
def has_any_pixels(mask: np.ndarray) -> bool:
    return int(np.count_nonzero(mask)) > 0


def has_large_component(mask: np.ndarray, min_area: int) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    return any(cv2.contourArea(c) >= min_area for c in contours)


def fill_black_holes_inside_white(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)
    inv = cv2.bitwise_not(padded)
    h, w = inv.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, ff_mask, (0, 0), 0)
    filled = cv2.bitwise_or(padded, inv)
    return filled[1:-1, 1:-1]


def whiten_black_near_border(mask: np.ndarray, border_px: int = 8) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape
    band = np.zeros_like(m, dtype=np.uint8)
    band[:border_px, :] = 255
    band[-border_px:, :] = 255
    band[:, :border_px] = 255
    band[:, -border_px:] = 255
    black = (m == 0).astype(np.uint8) * 255
    black_in_band = cv2.bitwise_and(black, band)
    contours, _ = cv2.findContours(black_in_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    to_whiten = np.zeros_like(m, dtype=np.uint8)
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        temp = black.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(temp, ff_mask, (x, y), 128)
        component = (temp == 128).astype(np.uint8) * 255
        to_whiten = cv2.bitwise_or(to_whiten, component)
    to_whiten = cv2.bitwise_and(to_whiten, band)
    m[to_whiten > 0] = 255
    return m


def detect_gums(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    red_hue = ((H <= 12) | (H >= 165))
    gum_mask = (red_hue & (S >= 60) & (V >= 60) & (A >= 150)).astype(np.uint8) * 255
    gum_mask = cv2.morphologyEx(gum_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    gum_mask = cv2.morphologyEx(gum_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8), iterations=1)
    gum_mask = fill_black_holes_inside_white(gum_mask)
    gum_mask = whiten_black_near_border(gum_mask, border_px=8)
    return gum_mask


def detect_plaque(rgb, tooth_mask, L_max, S_min, B_min, H_low, H_high):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    if H_low <= H_high:
        yellow_hue = (H >= H_low) & (H <= H_high)
    else:
        yellow_hue = (H >= H_low) | (H <= H_high)
    plaque = (yellow_hue & (S >= S_min) & (L <= L_max) & (B >= B_min)).astype(np.uint8) * 255
    plaque = cv2.bitwise_and(plaque, tooth_mask)
    plaque = cv2.morphologyEx(plaque, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    plaque = cv2.morphologyEx(plaque, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    h, w = plaque.shape
    vertical_mask = np.zeros_like(plaque, dtype=np.uint8)
    vertical_mask[int(h * 0.30):, :] = 255
    return cv2.bitwise_and(plaque, vertical_mask)


def detect_black_spots(rgb, tooth_mask, V_max, S_max, min_area,
                       overlap_ratio=0.80, ring_ratio=0.65, min_dist=3,
                       contrast_thresh=25.0):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    L_lab, A_lab, B_lab = cv2.split(lab)

    # 1. Global absolute darkness
    global_dark = (V.astype(np.int32) <= V_max).astype(np.uint8) * 255

    # 2. Local contrast — darker than neighbourhood
    L_float = L_lab.astype(np.float32)
    local_mean = cv2.GaussianBlur(L_float, (51, 51), 0)
    local_dark = ((local_mean - L_float) > float(contrast_thresh)).astype(np.uint8) * 255

    # 3. Brown / dark-brown colour cue (early-stage cavities)
    brown = (
        (H >= 5) & (H <= 25) &
        (S >= 40) & (S <= S_max) &
        (V >= 30) & (V <= int(V_max * 1.4))
    ).astype(np.uint8) * 255

    # Combine signals and restrict to tooth interior
    candidates = cv2.bitwise_or(cv2.bitwise_or(global_dark, local_dark), brown)
    candidates = cv2.bitwise_and(candidates, tooth_mask)
    inner_tooth = cv2.erode(tooth_mask, np.ones((7, 7), np.uint8), iterations=1)
    candidates = cv2.bitwise_and(candidates, inner_tooth)
    dist_map = cv2.distanceTransform((inner_tooth > 0).astype(np.uint8), cv2.DIST_L2, 5)

    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(candidates)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 3.5 * max(w, 1):
            continue
        if w / max(h, 1) > 5.0:
            continue
        if area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-5) < 0.35:
            continue
        mask_cnt = np.zeros_like(candidates)
        cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
        pixel_dists = dist_map[mask_cnt > 0]
        if len(pixel_dists) == 0 or np.max(pixel_dists) < min_dist:
            continue
        overlap_px = np.count_nonzero(cv2.bitwise_and(mask_cnt, inner_tooth))
        if overlap_px / (np.count_nonzero(mask_cnt) + 1e-5) < overlap_ratio:
            continue
        ring = cv2.subtract(cv2.dilate(mask_cnt, np.ones((11, 11), np.uint8), iterations=1), mask_cnt)
        ring_px = np.count_nonzero(ring)
        if ring_px > 0:
            if np.count_nonzero(cv2.bitwise_and(ring, inner_tooth)) / (ring_px + 1e-5) < ring_ratio:
                continue
        cv2.drawContours(keep, [cnt], -1, 255, -1)

    return keep


def detect_inflamed_gums(rgb, gum_mask, A_min, S_min, min_area):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    inflamed = ((A >= A_min) & (S >= S_min)).astype(np.uint8) * 255
    inflamed = cv2.bitwise_and(inflamed, gum_mask)
    inflamed = cv2.morphologyEx(inflamed, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    inflamed = cv2.morphologyEx(inflamed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(inflamed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(inflamed)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(keep, [cnt], -1, 255, thickness=-1)
    return keep


def detect_teeth(rgb, gum_mask):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    teeth = (
        (V >= 90) & (S <= 140) & (L >= 120) &
        (A >= 110) & (A <= 155) &
        (B >= 110) & (B <= 185)
    ).astype(np.uint8) * 255
    teeth = cv2.bitwise_and(teeth, cv2.bitwise_not(gum_mask))
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(teeth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(teeth)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 500:
            cv2.drawContours(keep, [cnt], -1, 255, -1)
    return keep


def draw_boundaries_and_label(rgb, mask, label, color_bgr=(0, 120, 255)):
    out = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 120]
    if not valid:
        return out
    for cnt in valid:
        cv2.drawContours(out, [cnt], -1, color_bgr, 3)
    x, y, w, h = cv2.boundingRect(max(valid, key=cv2.contourArea))
    cv2.putText(out, label, (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 3, cv2.LINE_AA)
    return out


# -----------------------------
# Main
# -----------------------------
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    rgb = np.array(img)

    gum_mask   = detect_gums(rgb)
    gums_found = has_any_pixels(gum_mask)

    tooth_mask = detect_teeth(
        rgb,
        gum_mask if gums_found else np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
    )

    plaque_mask = detect_plaque(
        rgb, tooth_mask,
        PLAQUE_L_MAX, PLAQUE_S_MIN, PLAQUE_B_MIN,
        PLAQUE_H_LOW, PLAQUE_H_HIGH
    )

    black_spot_mask = detect_black_spots(
        rgb, tooth_mask,
        BLACK_V_MAX, BLACK_S_MAX, MIN_BLACK_AREA,
        overlap_ratio=OVERLAP_RATIO, ring_ratio=RING_RATIO, min_dist=MIN_DIST,
        contrast_thresh=CONTRAST_THRESH
    )

    inflamed_mask = detect_inflamed_gums(
        rgb, gum_mask if gums_found else np.zeros_like(tooth_mask),
        INFLAMED_A_MIN, INFLAMED_S_MIN, MIN_INFLAMED_AREA
    )

    st.subheader("Original")
    st.image(rgb, width="stretch")

    # Plaque tab
    with tabs[0]:
        plaque_ok = not has_large_component(plaque_mask, MIN_BLACK_AREA)
        if plaque_ok:
            st.success("✅ No plaque detected. Looks good!")
        else:
            st.warning("⚠️ Plaque-like regions detected.")
        st.subheader("Plaque mask (gray)")
        st.image(np.where(plaque_mask > 0, 170, 0).astype(np.uint8), width="stretch")
        if show_plaque_overlay and not plaque_ok:
            st.subheader("Overlay (plaque boundary highlighted)")
            st.image(draw_boundaries_and_label(rgb, plaque_mask, "Plaque", (0, 120, 255)), width="stretch")

    # Cavity tab
    with tabs[1]:
        cavity_ok = not has_large_component(black_spot_mask, MIN_BLACK_AREA)
        if cavity_ok:
            st.success("✅ No cavities or dark spots detected. Looks good!")
        else:
            st.warning("⚠️ Dark spot regions detected (possible cavities).")
        st.subheader("Cavity Mask (gray)")
        st.image(np.where(black_spot_mask > 0, 170, 0).astype(np.uint8), width="stretch")
        if show_black_overlay and not cavity_ok:
            overlay = rgb.copy()
            contours, _ = cv2.findContours(black_spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) >= MIN_BLACK_AREA:
                    cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), thickness=3)
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.putText(overlay, "Cavity", (x, max(0, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            st.subheader("Cavity Highlight")
            st.image(overlay, width="stretch")

    # Gums tab
    with tabs[2]:
        if not gums_found:
            st.success("✅ Gums not detected in this image (may be a tooth-only photo).")
        else:
            inflamed_ok = not has_large_component(inflamed_mask, MIN_INFLAMED_AREA)
            if inflamed_ok:
                st.success("✅ No inflamed gums detected. Looks good!")
            else:
                st.warning("⚠️ Inflamed gum-like regions detected.")
        st.subheader("Inflamed mask (gray)")
        st.image(np.where(inflamed_mask > 0, 170, 0).astype(np.uint8), width="stretch")
        if show_inflamed_overlay and gums_found:
            if has_large_component(inflamed_mask, MIN_INFLAMED_AREA):
                st.subheader("Overlay (inflamed boundary highlighted)")
                st.image(draw_boundaries_and_label(rgb, inflamed_mask, "Inflamed", (0, 0, 255)), width="stretch")

else:
    st.info("Upload a tooth image to see results.")
