import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

lane_img = cv2.imread("./line.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(lane_img, cv2.COLOR_BGR2HSV)

lower_black = np.array([0, 0, 0])
upper_black = np.array([105, 105, 105])
mask = cv2.inRange(hsv, lower_black, upper_black)
canny = cv2.Canny(mask, 150, 300)

canny_img = np.copy(canny)
height = canny_img.shape[0]
width = canny_img.shape[1]

print(f"Image width : {width}, height : {height}\n")

## for middle locations
img = cv2.imread('./line.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_img = np.copy(gray)

d = height // 10
epo1 = 2*d
epo2 = d*9
crop_img = gray_img[epo1:epo2] # height를 d등분해서 3번째 (2d ~ 3d-1)

# strengthen
strengthen_img = np.power(crop_img, 2)
crop_height = strengthen_img.shape[0]
crop_width = strengthen_img.shape[1]
print(f"Crop image width : {crop_width}, height : {crop_height}\n")

middles = []

start = time.time()
for i in range(len(strengthen_img)):
    lcur = crop_width // 2 - (crop_width // 4)
    rcur = crop_width // 2 + (crop_width // 4)

    while(strengthen_img[i][lcur] == 0 and strengthen_img[i][rcur] == 0):
        if strengthen_img[i][lcur] == 0:
            lcur -= 1
        if strengthen_img[i][rcur] == 0:
            rcur += 1
        if lcur < 0:
            lcur = 0
            break
        if rcur > crop_width - 1:
            rcur = crop_width - 1
            break

    middle = (rcur + lcur) // 2
    middles.append(middle)

## draw middle line
color = (0, 255, 0)
for i in range(epo1, epo2):
    gray_img = cv2.line(gray_img,(middles[i-epo1],i), (middles[i-epo1],i), color, 5)

## plt.imshow(img)
plt.subplot(1, 2, 1)
plt.imshow(gray_img)
plt.subplot(1, 2, 2)
plt.imshow(crop_img)
plt.show()
x_center = np.mean(middles)

print(f"avg of middle point : {x_center}")
end = time.time()

print(f"running time : {end-start} sec")

# plt.imshow(strengthen_img)
# plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# lane_img = cv2.imread("./lane.jpg", cv2.IMREAD_COLOR)
# hsv = cv2.cvtColor(lane_img, cv2.COLOR_BGR2HSV)

# lower_black = np.array([0, 0, 0])
# upper_black = np.array([42, 60, 90])
# mask = cv2.inRange(hsv, lower_black, upper_black)
# canny = cv2.Canny(mask, 150, 300)

# print(canny.shape)
# f = open("./canny.txt", "w")
# s = ""
# for i, c in enumerate(canny):
#     s += (str(c) + "\n")
# f.write(s)
# f.close()

# plt.imshow(canny)
# plt.show()

########################################################################################

# imgs = []
# for i in range(3):
#     imgs.append(cv2.imread("./imgs/stair_%d.jpg" %(i+1), cv2.IMREAD_GRAYSCALE))
#     # imgs.append(cv2.imread("./imgs/stair_%d.jpg" %(i+1)))

# minVal = 100
# maxVal = 150

# canny = []
# for i in range(3):
# #     canny.append(cv2.Canny(imgs[i], 10 + i * 10, 50 * i * 10))
#     canny.append(cv2.Canny(imgs[i], minVal, maxVal))

# images = []
# for i in range(3):
#     images.append(imgs[i])
#     images.append(canny[i])
# titles = ['original', 'canny']

# for i in range(6):
#     plt.subplot(3, 2, i+1)
#     plt.imshow(images[i])
#     if i % 2 != 0:
#         plt.title(titles[i % 2] + "minVal(%d), maxVal(%d)" %(minVal, maxVal))
#     else:
#         plt.title(titles[i%2])
#     plt.xticks([])
#     plt.yticks([])
    
# plt.show()