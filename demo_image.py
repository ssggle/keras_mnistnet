from tensorflow.keras.models import load_model
import cv2
import imutils
from imutils.contours import sort_contours
import numpy as np
#获取图像
imgPath = "image/test3.jpg"
model_path = "model/model_mnist.h5"
is_show = True
# 获取视频
vs_img = cv2.imread(imgPath)
# 加载模型
model = load_model(model_path)
model.summary()

# 调整大小
frame = imutils.resize(vs_img,width=200)
# 调试过程中可以显示一下
# if Debug:
#     cv2.imshow("frame",frame)
#     cv2.waitKey(10)
# 转为灰度图
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# 滤波
bl = cv2.GaussianBlur(gray,(5,5),0)

# 边缘检测找轮廓
edge_canny = cv2.Canny(bl, 85, 200)

# 膨胀处理
kernel = np.ones((3,3),np.uint8)
edge_canny = cv2.dilate(edge_canny,kernel)


if is_show:
    cv2.imshow("edge_canny", edge_canny)
    cv2.waitKey(10)
items = cv2.findContours(edge_canny.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# 返回conts中的countors(轮廓)
conts = items[0] if len(items) == 2 else items[1]
# print(conts)
# if is_show:
#     cv2.drawContours(frame, conts, -1, (0, 255, 0), 1)
#
#     cv2.imshow("src",frame)
#     cv2.waitKey(10)
# 从左到右排序
conts,_ = sort_contours(conts,method="left-to-right")
# print(conts)
# 初始化列表放找到的字符
find_chars = []

#遍历找字符
for i in conts:
    #print(np.array(i))
    (x,y,w,h) = cv2.boundingRect(i)
    # 过滤一下，找出字符边框
    if(w>2 and w< 100) and (h>5 and h< 100):
        # 框字符
        roi = gray[y:y+h,x:x+w]
        mask = np.zeros(roi.shape,dtype="uint8")
        digit = cv2.bitwise_and(roi, roi, mask=mask)
        # 自动阈值处理
        _, th = cv2.threshold(roi, 0 ,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # 宽高
        th_H,th_W = th.shape
        # 缩放到28尺寸
        if th_H < th_W:
            th = imutils.resize(th,width=28)
        else:
            th = imutils.resize(th,height=28)
        # if is_show:
        #     cv2.imshow("th", th)
        #     cv2.waitKey(10)
        # 缩放后的宽高
        th_H, th_W = th.shape
        dx = int(max(0,28-th_W)/2)
        dy = int(max(0,28-th_H)/2)

        # 填充到28x28
        padding = cv2.copyMakeBorder(th,top=dy,bottom=dy,left=dx,right=dx,
                                     borderType=cv2.BORDER_CONSTANT,value=(0,0,0))
        padding = cv2.resize(padding,(28,28))

        # 缩放到0-1，扩展维度
        padding = padding.astype("float32")/255.0
        padding = np.expand_dims(padding,axis=-1)

        #存入列表
        print(((x,y,w,h)))
        find_chars.append((padding,(x,y,w,h)))
    else:
        print("next ... ")
        continue

# 提取
boxes = [b[1] for b in find_chars]

find_chars = np.array([f[0] for f in find_chars], dtype="float32")
if find_chars is None:
    print("can not find chars ...")

# 放入模型
predicts = model.predict(find_chars)

# 标签
labels = "0123456789"
# 预测显示
for (pred, (x,y,w,h)) in zip(predicts,boxes):
    # 返回最大值
    p = np.argmax(pred)
    pre = pred[p]
    label = labels[p]
    # 绘制框显示
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.putText(frame,label,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("result",frame)
    cv2.waitKey(10)


