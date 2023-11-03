import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read binary file and convert to numpy array
def read_binary_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    data = np.frombuffer(data, dtype=np.uint8)
    return data

def reshape_image(data, height, width):
    data = np.reshape(data, (height, width))
    return data

i1 = reshape_image(read_binary_file('camerabin.sec'), 256, 256)
i2 = reshape_image(read_binary_file('eyeRbin.sec'), 256, 256)
i3 = reshape_image(read_binary_file('headbin.sec'), 256, 256)
i4 = reshape_image(read_binary_file('salesmanbin.sec'), 256, 256)

# Compute DFT of i1, i2, i3, i4
i1_dft = np.fft.fft2(i1)
i2_dft = np.fft.fft2(i2)
i3_dft = np.fft.fft2(i3)
i4_dft = np.fft.fft2(i4)

# Display i1, i1_dft, i2, i2_dft, i3, i3_dft, i4, i4_dft
plt.subplot(431)
plt.imshow(i1, cmap='gray')
plt.title('i1')
plt.subplot(432)
plt.imshow(np.real(i1_dft), cmap='gray')
plt.title('i1_dft_real')
plt.subplot(433)
plt.imshow(np.imag(i1_dft), cmap='gray')
plt.title('i1_dft_imag')
plt.subplot(434)
plt.imshow(i2, cmap='gray')
plt.title('i2')
plt.subplot(435)
plt.imshow(np.real(i2_dft), cmap='gray')
plt.title('i2_dft_real')
plt.subplot(436)
plt.imshow(np.imag(i2_dft), cmap='gray')
plt.title('i2_dft_imag')
plt.subplot(437)
plt.imshow(i3, cmap='gray')
plt.title('i3')
plt.subplot(438)
plt.imshow(np.real(i3_dft), cmap='gray')
plt.title('i3_dft_real')
plt.subplot(439)
plt.imshow(np.imag(i3_dft), cmap='gray')
plt.title('i3_dft_imag')
plt.subplot(4,3,10)
plt.imshow(i4, cmap='gray')
plt.title('i4')
plt.subplot(4,3,11)
plt.imshow(np.real(i4_dft), cmap='gray')
plt.title('i4_dft_real')
plt.subplot(4,3,12)
plt.imshow(np.imag(i4_dft), cmap='gray')
plt.title('i4_dft_imag')
plt.show()

