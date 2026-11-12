"""HW06: HW05와 Lab05 Exercise 2 기반 자동 Object Counting.
핵심: Basic Global Thresholding(06장 28쪽), Median, Morphology, Connected Components.
참고: RGB 색 차이와 조건 분기는 사진 종류에 대응하기 위한 추가 전처리 아이디어임.
"""
import numpy as np
import scipy.ndimage as nd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st


def basic_global_threshold(image, eps=1.0):
    """Lab05 Exercise 2 / 06장 28p: 두 그룹의 평균으로 임계값 반복 계산."""
    x = np.asarray(image, dtype=np.float64)
    T = (float(x.min()) + float(x.max())) / 2
    for _ in range(100):
        G1 = x[x > T]
        G2 = x[x <= T]
        if len(G1) == 0 or len(G2) == 0:
            break
        T_new = (G1.mean() + G2.mean()) / 2
        if abs(T_new - T) < eps:
            T = T_new
            break
        T = T_new
    return T


def remove_small_objects(mask):
    """Retain connected regions that are large enough relative to the image."""
    label, _ = nd.label(mask)
    sizes = np.bincount(label.ravel())
    # HW05와 달리 이미지 크기에 맞춰 최소 픽셀 수를 정함
    minimum = max(8, round(mask.size * 0.000018))
    valid_areas = sizes[1:][sizes[1:] >= minimum]
    if len(valid_areas) >= 3:
        # 대다수 물체에 비해 지나치게 작은 점은 객체에서 제외
        minimum = max(minimum, round(float(np.median(valid_areas)) * 0.08))
    cleaned = (sizes[label] >= minimum) & (label != 0)
    # 지나치게 큰 배경 영역은 제외
    label, count = nd.label(cleaned)
    if count:
        sizes = np.bincount(label.ravel())
        cleaned &= (sizes[label] < mask.size * 0.55)
    return cleaned


def count_objects(image):
    """Return the intermediate masks and object-counting results."""
    image.thumbnail((900, 900))
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    # HW05의 Median Filter: 작은 낙엽도 살리기 위해 9x9 대신 3x3 사용
    rgb_smooth = nd.median_filter(rgb, size=(3, 3, 1))
    gray = np.asarray(Image.fromarray(rgb_smooth).convert('L'), dtype=np.uint8)
    h, w = gray.shape
    margin = max(2, min(h, w) // 35)
    edge = np.concatenate((
        rgb_smooth[:margin].reshape(-1, 3),
        rgb_smooth[-margin:].reshape(-1, 3),
        rgb_smooth[:, :margin].reshape(-1, 3),
        rgb_smooth[:, -margin:].reshape(-1, 3),
    )).astype(np.float32)
    reference = np.median(edge, axis=0)
    edge_distance = np.sqrt(np.sum((edge - reference) ** 2, axis=1))

    r = rgb_smooth[:, :, 0].astype(np.float32)
    g = rgb_smooth[:, :, 1].astype(np.float32)
    b = rgb_smooth[:, :, 2].astype(np.float32)
    yellowness = (r + g) / 2 - b

    # 노란색 물체가 두드러지면 R/G와 B의 차이를 사용
    if np.percentile(yellowness, 99) > 37 and np.percentile(yellowness, 80) < 54 and np.percentile(edge_distance, 75) > 27:
        score = np.clip(yellowness, 0, 255)
        threshold = max(13, basic_global_threshold(score))
        initial = score > threshold
        method = '색상 차이 + Basic Global Thresholding'
    elif np.percentile(edge_distance, 75) < 60:
        delta = np.sqrt(np.sum((rgb_smooth.astype(np.float32) - reference) ** 2, axis=2))
        score = np.clip(delta, 0, 255)
        threshold = max(int(np.percentile(edge_distance, 75) * 1.5), 12, basic_global_threshold(score))
        initial = score > threshold
        method = '배경 색상 차이 + Basic Global Thresholding'
    else:
        # 배경 밝기가 균일하지 않으면 Gaussian Filter로 배경을 근사
        smoothed = gray.astype(np.float32)
        bg = nd.gaussian_filter(smoothed, sigma=max(9, min(h, w) / 38))
        local_bright = np.maximum(smoothed - bg, 0)
        local_dark = np.maximum(bg - smoothed, 0)
        # 영상 가장자리를 덜 차지하는 쪽을 물체 후보로 선택
        candidates = []
        for candidate in (local_bright, local_dark):
            t = max(11, basic_global_threshold(candidate))
            mask = candidate > t
            edge_count = np.concatenate((mask[:margin].ravel(), mask[-margin:].ravel(), mask[:, :margin].ravel(), mask[:, -margin:].ravel())).mean()
            candidates.append((edge_count, mask, t))
        _, initial, threshold = min(candidates, key=lambda x: x[0])
        method = '국소 밝기 차이 + Basic Global Thresholding'

    # Closing: 작은 틈을 메움 (2x2 SE, 최근 Morphology 수업에서 배운 연산)
    initial = nd.binary_closing(initial, structure=np.ones((2, 2)), border_value=0)
    cleaned = remove_small_objects(initial)
    labels, number = nd.label(cleaned)
    return rgb, gray, initial, cleaned, labels, number, method, threshold


st.title('HW06 - Object Counting')
st.write('사진을 업로드하면 자동으로 물체의 개수를 계산합니다.')

uploaded = st.file_uploader('JPG 또는 PNG 사진 업로드', type=['jpg', 'jpeg', 'png'])
if uploaded is not None:
    rgb, gray, initial, clean, labels, number, method, threshold = count_objects(Image.open(uploaded))
    st.subheader(f'검출된 영역의 개수: {number}')
    st.caption(f'자동 분석 방식: {method} / 임계값: {threshold:.1f}')

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title('Input image')
    ax[0, 1].imshow(gray, cmap='gray', vmin=0, vmax=255)
    ax[0, 1].set_title('Median filtered')
    ax[1, 0].imshow(clean, cmap='gray')
    ax[1, 0].set_title('Final binary mask')
    ax[1, 1].imshow(labels, cmap='nipy_spectral')
    ax[1, 1].set_title('Connected components')
    for a in ax.flat:
        a.axis('off')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if number:
        sizes = np.bincount(labels.ravel())[1:]
        biggest = int(np.argmax(sizes)) + 1
        smallest = int(np.argmin(sizes)) + 1
        fig2, ax2 = plt.subplots(1, 2, figsize=(9, 4))
        ax2[0].imshow(labels == biggest, cmap='gray')
        ax2[0].set_title('Largest component')
        ax2[1].imshow(labels == smallest, cmap='gray')
        ax2[1].set_title('Smallest component')
        for a in ax2:
            a.axis('off')
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
