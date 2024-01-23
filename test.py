from tensorflow import keras
import numpy as np
#Keras 라이브러리를 사용하여 MNIST 데이터셋을 로드
(train_input, train_target), (test_input, test_target) = keras.datasets.mnist.load_data()

train_scaled =  train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
print(train_scaled.shape)

from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

sc =  SGDClassifier(loss="log_loss", max_iter=5, random_state=42)

scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(scores)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import train_test_split
# scikit-learn의 train_test_split 함수를 사용하여 데이터를 학습 세트와 검증 세트로 분할합니다
train_scaled, val_scaled, train_target, val_target = train_test_split(
        train_scaled, train_target, test_size = 0.2, random_state=42)
# 학습과 검증 데이터와 레이블의 크기를 출력합니다.
print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)
# 케라스의 Sequential 모델을 생성합니다.
model = keras.Sequential()
# 신경망을 구성해줌
model.add(Flatten(input_shape=(28*28,)))
model.add(Dense(units=10, activation="softmax"))
# model이 어떻게 동작하는지 지정
model.compile(optimizer=SGD(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy'] )
# 설정이 다되었으면 모델 학습
history = model.fit(train_scaled, train_target, epochs=100, verbose=1, validation_data=(val_scaled, val_target))
test_scaled =  test_input / 255.0
test_scaled = test_scaled.reshape(-1, 28*28)
print(test_scaled.shape)
# Evaluation

print(model.evaluate(test_scaled, test_target))

#       loss                accuracy
# [0.2931129038333893, 0.9175999760627747]
#                      0.8919166666666666  머신러닝

# 우리 모델은 정확도가 91.7%인 모델이예요!
# 머신러닝의 Regression 중 Logistic Regression을 여러개 결합해서
# 구현한 Multinomial 구현이예요!