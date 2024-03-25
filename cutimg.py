import numpy as np
import cv2
import matplotlib.pyplot as plt

def cutimg(img,num,overlap_factor):   
    '''img 是图像矩阵，num是切过后子图数目，因为统一为正方形切图，因此这里的num开方后需要为整数，
        overlap_factor 是切分后重叠部分的步长'''
    factor = int(np.sqrt(num))
    rawshape = max(img.shape)
    cutrawshape = rawshape // (factor)
    resizeshape = int(cutrawshape // 32) * 32    # 因为模型要求最小送入矩阵为32
    img = cv2.resize(img, (factor*resizeshape, factor*resizeshape))
    img_stacks = []    # 返回结果装载矩阵
    overlap_factor = overlap_factor
    cutshape = int((factor*resizeshape+overlap_factor)/factor)   # 需要保证除以factor整除
    for i in range(factor):
        for ii in range(factor):
            img_temp = img[(ii*cutshape-ii*overlap_factor):((ii+1)*(cutshape)-ii*overlap_factor),
                           (i*cutshape-i*overlap_factor):((i+1)*cutshape-i*overlap_factor)]
            img_stacks.append(img_temp)
 
    return img_stacks

img1 = cv2.imread('Image_20231121103732690.bmp')
# 指定切分的子图数目和重叠因子
num = 4
overlap_factor = 10

# 调用 cutimg 函数分割第一张图像
img1_stacks = cutimg(img1, num, overlap_factor)
 
# img1_stacks = cutimg(img1, num, overlap_factor)

# 可视化切分后的图像块
fig, axes = plt.subplots(num, num, figsize=(10, 10))
for i in range(num):
    for j in range(num):
        axes[i, j].imshow(cv2.cvtColor(img1_stacks[i * num + j], cv2.COLOR_BGR2RGB))  # 注意 OpenCV 读取的图像格式是 BGR，需转换为 RGB
        axes[i, j].axis('off')
plt.show()