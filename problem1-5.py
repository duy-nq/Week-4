import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

# Create a 8x8 array that hold complex number
i1 = np.zeros((8,8), dtype=np.complex64)
i2 = np.zeros((8,8), dtype=np.complex64)
i3 = np.zeros((8,8), dtype=np.complex64)
i4 = np.zeros((8,8), dtype=np.complex64)
i5 = np.zeros((8,8), dtype=np.complex64)

# Assign value to i1
for row in range(8):
    for col in range(8):
        tmp = row*2+col*2
        i1[row][col] = 0.5 * (math.cos(math.pi*tmp*0.25)+1j*math.sin(math.pi*tmp*0.25))

# Assign value to i2
for row in range(8):
    for col in range(8):
        tmp = row*2+col*2
        i2[row][col] = 0.5 * (math.cos(math.pi*tmp*0.25)-1j*math.sin(math.pi*tmp*0.25))

# Assign value to i3
for row in range(8):
    for col in range(8):
        tmp = row*2+col*2
        i3[row][col] = math.cos(math.pi*tmp*0.25)

# Assign value to i4
for row in range(8):
    for col in range(8):
        tmp = row*2+col*2
        i4[row][col] = 0.5 * (math.cos(math.pi*tmp*0.25)+1j*math.sin(math.pi*tmp*0.25))


# Assign value to i5
for row in range(8):
    for col in range(8):
        tmp = row*2+col*2
        i5[row][col] = 0.5 * (math.cos(math.pi*tmp*0.25)+1j*math.sin(math.pi*tmp*0.25))

# Compute the 2D DFT of the image
i1_dft = np.fft.fft2(i1)
i2_dft = np.fft.fft2(i2)
i3_dft = np.fft.fft2(i3)
i4_dft = np.fft.fft2(i4)
i5_dft = np.fft.fft2(i5)

# Displat the i2_dft
plt.subplot(231)
plt.imshow(np.abs(i1_dft), cmap='gray')
plt.title('i1_dft')
plt.subplot(232)
plt.imshow(np.abs(i2_dft), cmap='gray')
plt.title('i2_dft')
plt.subplot(233)
plt.imshow(np.abs(i3_dft), cmap='gray')
plt.title('i3_dft')
plt.subplot(234)
plt.imshow(np.abs(i4_dft), cmap='gray')
plt.title('i4_dft')
plt.subplot(235)
plt.imshow(np.abs(i5_dft), cmap='gray')
plt.title('i5_dft')
plt.show()





