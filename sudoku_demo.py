# 命令行
# python sudoku_demo.py --model model/model_mnist.h5 --image image/shudu01.jpg

from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2

def extract_number(image, is_show=False):
	# 自动阈值
	th = cv2.threshold(image, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	th = clear_border(th)

	if is_show:
		cv2.imshow("image th...", th)
		cv2.waitKey(0)

	# 在小方格里找数字轮廓,获取轮廓
	cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# 判断
	if len(cnts) == 0:
		return None

	# 找出最大轮廓，并绘制
	cont_max = max(cnts, key=cv2.contourArea)
	mask = np.zeros(th.shape, dtype="uint8")
	cv2.drawContours(mask, [cont_max], -1, 255, -1)

	# 通过轮廓占比（mask像素相对总面积）去噪声，或去除无用轮廓
	(h, w) = th.shape
	percent_fill = cv2.countNonZero(mask) / float(w * h)
	if percent_fill < 0.04:
		return None

	# 图像与操作
	number = cv2.bitwise_and(th, th, mask=mask)

	if is_show:
		cv2.imshow("number", number)
		cv2.waitKey(0)
	return number



ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained digit classifier")
ap.add_argument("-i", "--image", required=True,
	help="path to input sudoku puzzle image")
ap.add_argument("-d", "--is_show", type=int, default=-1,
	help="is show each step ")
args = vars(ap.parse_args())

# 加载模型
model = load_model(args["model"])
print("loading digit classifier...")
# 获取图像
image = cv2.imread(args["image"])
print("processing image...")
if image is None:
	print("could not load image ...")
# 调整大小
image = imutils.resize(image, width=400)
src = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# 9x9 数独格板
board_9 = np.zeros((9, 9), dtype="int")

# 宽度和高度方向上单个小方格尺寸
stepX = gray.shape[1] // 9
stepY = gray.shape[0] // 9

# 存每个小格子位置
each_loc = []

# 获取格子位置
for y in range(0, 9):
	# 当前格子位置
	c_row = []

	for x in range(0, 9):
		# 当前格子坐标
		startX = x * stepX
		startY = y * stepY
		endX = (x + 1) * stepX
		endY = (y + 1) * stepY
		# 存下来
		c_row.append((startX, startY, endX, endY))

		# 拿到小格子，并提取数字
		grid_img = gray[startY:endY, startX:endX]
		number = extract_number(grid_img, is_show=False)

		# 判断一下
		if number is not None:
			two_h = np.hstack([grid_img, number])
			# cv2.imshow("grid_img/number", two_h)

			# 将格子图缩放到28x28
			roi = cv2.resize(number, (28, 28))
			roi = roi.astype("float") / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)

			# 预测格子里的数字
			pred = model.predict(roi).argmax(axis=1)[0]
			board_9[y, x] = pred

	# 放入列表
	each_loc.append(c_row)

# 数独板并显示
print("OCR sudoku board:")
makeup = Sudoku(3, 3, board=board_9.tolist())
makeup.show()

# 计算填写
print("solving sudoku makeup...")
solution = makeup.solve()
solution.show_full()

# 遍历每个格子
for (grid, b) in zip(each_loc, solution.board):

	for (box, n) in zip(grid, b):
		# 坐标位置
		startX, startY, endX, endY = box

		# 显示信息
		textX = int((endX - startX) * 0.3)
		textY = int((endY - startY) * -0.25)
		textX += startX
		textY += endY

		cv2.putText(src, str(n), (textX, textY),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Results", src)
cv2.waitKey(0)
cv2.imwrite("output/res.jpg",src)
