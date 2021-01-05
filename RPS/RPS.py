import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# OOM 방지
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()


    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.03,
        height_shift_range=0.03,
        shear_range=0.03,
        zoom_range=0.03,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
 


    training_generator = training_datagen.flow_from_directory(TRAINING_DIR, 
                                                          batch_size=80,
                                                          target_size=(150, 150),
                                                          class_mode='categorical',                                                       
                                                          subset='training',
                                                         )

    validation_generator = training_datagen.flow_from_directory(TRAINING_DIR, 
                                                            batch_size=80,
                                                            target_size=(150, 150),
                                                            class_mode='categorical',
                                                            subset='validation', 
                                                            )
    
    model = tf.keras.models.Sequential([
         # Conv2D, MaxPooling2D 조합으로 층을 쌓습니다. 첫번째 입력층의 input_shape은 (150, 150, 3)으로 지정
        Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),  #64는 필터의 개수 (3,3)은 필터의 사이즈 
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        # 2D -> 1D로 변환을 위한 Flatten 
        Flatten(),
        # 과적합 방지를 위한 Dropout을 적용
        Dropout(0.5),
        # Dense layer
        Dense(512, activation='relu'), # 512는 노드의 개수
                                        
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    # 체크포인트 생성
    checkpoint_path = "my_ckpt.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                save_weights_only=True,
                                save_best_only=True,
                                monitor='val_loss',
                                 verbose=1
                                )
    # 모델 컴파일                            
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])  
    
    # 모델 학습
    model.fit(training_generator,
                validation_data=(validation_generator),
                epochs=25,
                callbacks=[checkpoint],
                )
    
    # 체크 포인트에 저장된 가중치로 업데이트
    model.load_weights(checkpoint_path)
    return model

# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you

if __name__ == '__main__':
    model = model()  # 모델 실행
    model.save("rps.h5")  # 학습된 파라미터 저장
