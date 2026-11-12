import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image
import streamlit as st

st.title('HW06 - Object Counting')
#st.write('HW05에서 사용한 Median Filter, Thresholding, Connected Component를 이용합니다.')

# 사진은 웹사이트에서 직접 업로드해서 사용
uploaded_file = st.file_uploader('분석할 사진을 업로드하세요.', type=['jpg', 'jpeg', 'png'])

# HW05에서 고정했던 값들을 사진에 맞게 조절할 수 있도록 수정
st.subheader('영상처리 설정')
scale = st.selectbox('이미지 축소 간격 (1: 원본, 2: 절반)', [1, 2, 4], index=1)
filter_size = st.select_slider('Median Filter 크기', options=[1, 3, 5, 7, 9], value=9)
th_val = st.slider('Threshold', 0, 255, 140)
direction = st.radio('검출할 물체의 밝기', ['배경보다 밝은 물체 (>)', '배경보다 어두운 물체 (<)'], horizontal=True)
min_area = st.slider('최소 객체 크기 (픽셀)', 0, 3000, 1000, 10)

if uploaded_file is not None:
    # 1. grayscale 이미지 읽기 (HW05의 [::2, ::2]를 조절 가능하게 변경)
    I = Image.open(uploaded_file).convert('L')
    f = np.array(I)[::scale, ::scale]

    # 2. Median filtering
    f_denoised = signal.medfilt2d(f, filter_size) if filter_size > 1 else f.copy()

    # 3. Thresholding: 사진에 따라 이진화 방향 선택
    if direction == '배경보다 밝은 물체 (>)':
        f_mask = f_denoised > th_val
    else:
        f_mask = f_denoised < th_val

    # 4. 작은 영역 제거 후 Connected Component Counting
    labels, nums = nd.label(f_mask)
    sizes = np.bincount(labels.ravel())
    f_mask_clean = sizes[labels] >= min_area
    f_mask_clean[labels == 0] = False
    labels, nums = nd.label(f_mask_clean)

    st.subheader(f'검출된 영역의 개수: {nums}')
    if nums == 0:
        st.warning('검출된 물체가 없습니다. Threshold, 최소 객체 크기 또는 이진화 방향을 바꿔보세요.')

    # 처리 단계별 이미지 출력 (HW05와 동일한 구성)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(f, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Input image')
    axes[0, 1].imshow(f_denoised, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Median filtered')
    axes[1, 0].imshow(f_mask, cmap='gray')
    axes[1, 0].set_title('Binary mask')
    axes[1, 1].imshow(labels, cmap='Spectral_r')
    axes[1, 1].set_title('Connected components')
    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 최종 분할 영상도 보여줘서 작은 영역 제거 효과를 비교
    st.write('작은 영역을 제거한 최종 Binary Mask')
    st.image((f_mask_clean.astype(np.uint8) * 255), clamp=True)

    # 5. 가장 큰 영역과 작은 영역
    if nums > 0:
        cc = np.zeros((nums, 2))
        for i in range(1, nums + 1):
            cc[i - 1, 0] = i
            cc[i - 1, 1] = np.sum(labels == i)

        ind_max = np.argmax(cc[:, 1])
        ind_min = np.argmin(cc[:, 1])

        fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
        axes2[0].imshow(f, cmap='gray', vmin=0, vmax=255)
        axes2[0].set_title('Input image')
        axes2[1].imshow(labels == cc[ind_max, 0], cmap='gray')
        axes2[1].set_title('Largest component')
        axes2[2].imshow(labels == cc[ind_min, 0], cmap='gray')
        axes2[2].set_title('Smallest component')
        for ax in axes2:
            ax.axis('off')
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
