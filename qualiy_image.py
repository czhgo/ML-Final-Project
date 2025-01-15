from keras.applications import VGG16
from keras.preprocessing import image
import numpy as np

# 加载预训练的VGG16模型
model = VGG16(weights='imagenet', include_top=False)

# 准备图像
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x)

# 提取特征
features = model.predict(x)

# 假设我们使用特征的第一个值作为质量评分（这只是一个示例）
quality_score = features[0][0]

print(f"Image quality score: {quality_score}")