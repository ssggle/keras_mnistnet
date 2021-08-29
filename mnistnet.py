from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

class MnistNet:
	@staticmethod
	def build(width, height, depth, classes):
		# 初始化模型
		model = Sequential()
		inputShape = (height, width, depth)

		# 从 CONV 到 RELU 到 POOL layers
		model.add(Conv2D(32, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# 再次从 CONV 到 RELU 到 POOL layers
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# FC层到relu层
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# 再次FC层到relu层
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# 用softmax函数分类
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# 返回模型
		return model