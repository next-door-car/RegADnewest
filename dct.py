import torch
import numpy as np


import numpy as np
import cv2


# 读取图像
image = cv2.imread('Image_20231121103732690.bmp', cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像
image = cv2.resize(image, (256, 256))

# 确保图像的尺寸是 8 的倍数，如果需要可以进行裁剪
height, width = image.shape[:2]
new_height = height - height % 16
new_width = width - width % 16
image = image[:new_height, :new_width]

# 将图像分块为 8x8 的小块，并对每个块应用 DCT 变换
dct_blocks = []
for y in range(0, new_height, 16):
    for x in range(0, new_width, 16):
        block = image[y:y+16, x:x+16]
        image_float32 = block.astype('float32')
        # image_umat = cv2.UMat(image_float32)
        dct_block = cv2.dct(image_float32)
        dct_blocks.append(dct_block)

# 将 DCT 变换后的结果拼接成一个图像
dct_image = np.zeros((new_height, new_width), dtype=np.float32)
for i, dct_block in enumerate(dct_blocks):
    y = (i // (new_width // 16)) * 16
    x = (i % (new_width // 16)) * 16
    dct_image[y:y+16, x:x+16] = dct_block

# 显示 DCT 变换后的结果图像
cv2.imshow('DCT Image', dct_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image = np.arange(1, 65).reshape(8, 8)
# image = np.float32(image)
# print("原始图像块数据:\n", image)
# image_float32 = image.astype('float32')
# image_umat = cv2.UMat(image_float32)
# dct_array = cv2.dct(image_umat)
# print("DCT变换结果:\n", dct_array)


# #  1.构造图像块伪数据 
# image = torch.arange(1, 65).reshape(1, 1, 8, 8)
# image_dct_int = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
# image_dct = image_dct_int.float()
# print("原始图像块:\n", image_dct)

# # 2. 构造DCT变换系数
# coff = torch.zeros((8, 8), dtype=torch.float)
# coff[0, :] = 1 * np.sqrt(1 / 8)
# for i in range(1, 8):
#     for j in range(8):
#         coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

# # 3. 执行DCT变换
# image_dct = torch.matmul(coff, image_dct)
# coff_permute = coff.permute(1, 0)
# image_dct1 = torch.matmul(image_dct, coff_permute)

# image_dct2 = torch.cat(torch.cat(image_dct1.chunk(1, 0), 3).chunk(1, 0), 2)
# print("DCT变换后的结果:\n", image_dct2)
