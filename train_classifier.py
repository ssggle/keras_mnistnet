# 命令行输入
# python train_classifier.py --model model/model_mnist.h5

from mnistnet import MnistNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

# 构造参数
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="model output path")
args = vars(ap.parse_args())

# 设定学习率，迭代次数，送入网络批次大小
INIT_LR = 1e-3
EPOCHS = 16
Batch_Size = 160

# 获取MNIST dataset
print("[LOGS] Please wait...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# 训练数据和测试数据设定维度
print(trainData.shape[0])
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# 归一化0-1之间
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# 标签转为向量
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

# 初始化模型，如果只识别两类则loss = “ binary_crossentropy”
opt = Adam(lr=INIT_LR)
model = MnistNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print("[LOGS] compiling model...")

# 训练
H = model.fit(
	trainData, trainLabels,
	validation_data=(testData, testLabels),
	batch_size=Batch_Size,
	epochs=EPOCHS,
	verbose=1)
print("[LOGS] training network...")

# 评估网络模型
predictions = model.predict(testData)
print("[LOGS] evaluating network...")
print(classification_report(
	testLabels.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))

# 保存模型
model.save(args["model"], save_format="h5")