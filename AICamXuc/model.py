from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, GlobalAveragePooling1D, Softmax, Multiply, LSTM, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Các lớp CNN
    x = Conv1D(32, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(64, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Lớp LSTM
    lstm1_out = LSTM(64, return_sequences=True)(x)
    lstm2_out = LSTM(128, return_sequences=True)(lstm1_out)
    
    # Cơ chế chú ý
    attention = Dense(1, activation='relu')(lstm2_out)
    attention = Softmax(axis=1)(attention)
    attended = Multiply()([lstm2_out, attention])
    x = GlobalAveragePooling1D()(attended)
    
    # Lớp đầu ra
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model