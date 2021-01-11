import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def model():
    
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()

    x_train = x_train / 255.0
    x_valid = x_valid / 255.0
    tf.keras.backend.set_floatx('float64')
    x = Flatten(input_shape=(28, 28))

    model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten으로 shape 펼치기
    
    # Dense Layer
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax'),  # Classification을 위한 Softmax
])
                        
    model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['acc'])  # 모델 컴파일

    # 체크포인트 생성
    checkpoint_path = "my_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', 
                             verbose=1)
    # 모델 학습
    history = model.fit(x_train, y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=30,
                    callbacks=[checkpoint],
                   )
                   
    # 가중치 업데이트    
    model.load_weights(checkpoint_path)

    return model


if __name__ == '__main__':
    model = model()
    model.save("fashion-mnist.h5")

# 결과 : val_loss: 0.3305 - val_acc: 0.8863
