import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[80%:]')

# 전처리 과정
def preprocess(data):
    x = data['features']
    y = data['label']
    y = tf.one_hot(y, 3)  # One-hot encoding
    return x, y


def model():
    train_data = train_dataset.map(preprocess).batch(10)
    valid_data = valid_dataset.map(preprocess).batch(10)
    

    # Dense Layer
    model = tf.keras.models.Sequential([
    Dense(256, activation='relu', input_shape=(4,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax'), # 3가지로 분류하기 위한 softmax 함수 사용
    ])
    
    # 모델 컴파일
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    # 체크포인트 생성
    checkpoint_path = "my_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                 save_weights_only=True, 
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)
    
    # 모델 학습
    model.fit(train_data,
              validation_data=(valid_data),
              epochs=20,
              callbacks=[checkpoint],
              )
    
    # 가중치 업데이트
    model.load_weights(checkpoint_path)
    
    return model


if __name__ == '__main__':
    model = model()
    model.save("iris.h5")
# 결과: val_loss: 0.1260 val_acc: 0.9667    
    
    
