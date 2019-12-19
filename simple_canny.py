import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

img = cv2.imread('./line.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize=3)
crop_img = np.copy(edges)

epo1 = 0
epo2 = crop_img.shape[0]

# strengthen
strengthen_img = np.power(crop_img, 2)
crop_height = strengthen_img.shape[0]
crop_width = strengthen_img.shape[1]
print(f"Crop image width : {crop_width}, height : {crop_height}\n") # 3024, 4032

middles = []

cnt = 0 ### for test

start = time.time()
for i in range(crop_height):
    # lcur = crop_width // 2 - (crop_width // 4)
    # rcur = crop_width // 2 + (crop_width // 4)
    lcur = crop_width // 2 - 1
    rcur = crop_width // 2 + 1
    prev_lcur = 0
    prev_rcur = 0

    while(strengthen_img[i][lcur] == 0 and strengthen_img[i][rcur] == 0):
        if strengthen_img[i][lcur] == 0:
            lcur -= 1
        if strengthen_img[i][rcur] == 0:
            rcur += 1
        if lcur < 0:
            lcur = prev_lcur
            break
        if rcur > crop_width - 1:
            rcur = prev_rcur
            break
        prev_lcur = lcur
        prev_rcur = rcur

    if cnt == 0: ### for test
        print(lcur)
        print(rcur)
        print((rcur - lcur)//2)
        print(strengthen_img[0][lcur])
        print(strengthen_img[0][rcur])
        cnt += 1

    middle = (rcur + lcur) // 2
    middles.append(middle)
# print(f"middles shape :{np.array(middles).shape}\n")


color = (255, 0, 0)
for i in range(epo1, epo2):
    # print(middles[i-epo1])
    crop_img = cv2.line(crop_img, (middles[i-epo1], i), (middles[i-epo1], i), color, 5)

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',crop_img)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

plt.imshow(crop_img)
plt.show()
end = time.time()

print(f"running time : {end-start} sec")