import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image
import streamlit as st

st.title('HW06 - Object Counting')
st.write('물체의 개수를 계산합니다.')

uploaded_file = st.file_uploader('분석할 사진을 업로드하세요.', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 1. HW05와 동일하게 grayscale로 변환하고 크기를 1/2로 줄임
    I = Image.open(uploaded_file).convert('L')
    f = np.array(I)[::2, ::2]

    # 2. Median filtering
    f_denoised = signal.medfilt2d(f, 9)

    # 3. Thresholding
    th_val = 140
    f_mask = f_denoised > th_val

    # 4. 작은 영역 제거 후 connected component 계산
    labels, nums = nd.label(f_mask)
    sizes = np.bincount(labels.ravel())
    f_mask_clean = sizes[labels] >= 1000
    f_mask_clean[labels == 0] = False
    labels, nums = nd.label(f_mask_clean)

    st.subheader(f'검출된 영역의 개수: {nums}')

    # 수업 실습처럼 처리 과정의 영상을 각각 출력
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(f, cmap='gray')
    axes[0, 0].set_title('Input image')
    axes[0, 1].imshow(f_denoised, cmap='gray')
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

    # 5. 가장 큰 영역과 작은 영역 (물체가 0개일 때도 오류가 나지 않도록)
    if nums > 0:
        cc = np.zeros((nums, 2))
        for i in range(1, nums + 1):
            cc[i - 1, 0] = i
            cc[i - 1, 1] = np.sum(labels == i)

        ind_max = np.argmax(cc[:, 1])
        ind_min = np.argmin(cc[:, 1])

        fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
        axes2[0].imshow(f, cmap='gray')
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
