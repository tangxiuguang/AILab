import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 设置可见设备（禁用CPU，只启用第一个GPU）
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[1], 'GPU')  # 使用第一个GPU

# 验证设备
print("当前使用的设备:", tf.config.list_logical_devices())

def load_mnist_local(path='mnist.npz'):
    """从本地加载MNIST数据集"""
    with np.load(path) as data:
        X_train = data['x_train']
        y_train = data['y_train']
        X_test = data['x_test']
        y_test = data['y_test']
    return (X_train, y_train), (X_test, y_test)

# 加载数据
(X_train, y_train), (X_test, y_test) = load_mnist_local('../../../datasets/mnist/mnist.npz')

# 数据预处理（仅一次！）
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 检查标签格式（确保是整数，非one-hot）
print("标签示例:", y_train[0])  # 应输出如 `5`

# 构建模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # 防止过拟合
    layers.Dense(10, activation='softmax')
])

# 编译模型（使用Adam优化器）
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# 评估
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# 可视化预测
predictions = model.predict(X_test)
predicted_labels = tf.argmax(predictions, axis=1)

plt.figure(figsize=(10, 4))
for i in range(10):
    idx = np.random.randint(0, len(X_test))
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    plt.title(f'True: {y_test[idx]}\nPred: {predicted_labels[idx].numpy()}')
    plt.axis('off')
plt.tight_layout()
plt.show()
