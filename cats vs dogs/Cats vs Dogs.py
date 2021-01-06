import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

dataset_name = 'cats_vs_dogs'

# 처음 80%의 데이터를 train_data로 사용
train_dataset = tfds.load(name=dataset_name, split='train[:80%]')  # tfds를 사용하여 데이터를 load한다.

# 최근 20%의 데이터를 validation_data로 사용
valid_dataset = tfds.load(name=dataset_name, split='train[80%:]')  # tfds를 사용하여 데이터를 load한다.

# 전처리
def preprocess(data):
    x = data['image'] / 255  # 이미지 정규화
    y = data['label']
    x = tf.image.resize(x, size=(224, 224))  # load한 모든 이미지를 224x224 사이즈로 변경시킨다.
    return x, y


def model():
    train_data = train_dataset.map(preprocess).batch(32)  # batch_size 32로 train_dataset을 전처리한다.
    valid_data = valid_dataset.map(preprocess).batch(32)  # batch_size 32로 valid_dataset을 전처리한다.
    
    #  CNN 모델링
    model = Sequential([
        Conv2D(32, (3,3), activation='relu',input_shape=(224, 224, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(512, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),                  
        Dense(2, activation='softmax')
    ])
    
    model.summary()
    
    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    
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
              epochs=15,
              callbacks=[checkpoint],
              )    
    
    # 가중치 업데이트
    model.load_weights(checkpoint_path)    
    
    return model


if __name__ == '__main__':
    model = model()
    model.save("cats-vs-dogs.h5")  # 학습된 파라미터 파일로 저장
    
# 결과 : val_loss:0.33644, val_acc:0.8637    
