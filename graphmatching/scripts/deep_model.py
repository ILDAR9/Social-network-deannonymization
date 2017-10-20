from keras.models import Model
from keras.layers import Dense, Lambda, Input, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.normalization import BatchNormalization
import keras

bin = 42
def get_model(feature_amount,
              best_weights_filepath='./best_weights.hdf5',
              log_path='./logs_nn/'):
    '''hyperparams'''
    #     learning_rate = 1e-3
    #     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
    #     optimiser = Adam(lr=learning_rate)
    dense_activation = 'relu'
    dense_1layer_nn_count_coef = 1
    dense_2layer_nn_count_coef = 1

    out_len = 1
    others_count = feature_amount - bin * 2
    in_shape = (feature_amount,)

    ''' Input '''
    inputs = Input(shape=in_shape, name="in_data")
    inputs_transformed = BatchNormalization()(inputs)

    ''' Slices '''
    input_bins = [None] * bin
    for i in range(bin):
        slice_0 = Lambda(lambda x: x[:, i:i + 1],
                         output_shape=lambda in_shape: (in_shape[0], 1))(inputs_transformed)
        slice_1 = Lambda(lambda x: x[:, bin + i:bin + i + 1],
                         output_shape=lambda in_shape: (in_shape[0], 1))(inputs_transformed)
        input_bins[i] = keras.layers.concatenate([slice_0, slice_1])

    slice_others = Lambda(lambda x: x[:, -others_count:],
                          output_shape=lambda in_shape: (in_shape[0], feature_amount - bin * 2))(inputs_transformed)

    ''' Dense for input_bins '''
    pair_denses = [None] * bin
    for i in range(bin):
        input_bins[i]
        pair_denses[i] = Dense(units=1, activation=dense_activation)(input_bins[i])

    ''' Dense others '''
    dense_others = Dense(units=others_count, activation=dense_activation)(slice_others)

    ''' Combine Denses '''
    combine_pairs_and_others = keras.layers.concatenate(pair_denses + [dense_others])
    dense_nn_count = bin + others_count
    dropout = Dropout(0.2)(combine_pairs_and_others)
    dense_model = Dense(units=dense_nn_count, activation=dense_activation)(dropout)

    ''' Final Denses '''
    dropout = Dropout(0.1)(dense_model)
    dense_nn_count = bin // 2 + others_count // 2
    dense_model = Dense(units=dense_nn_count, activation=dense_activation)(dense_model)
    output = Dense(units=out_len, activation='sigmoid', name='out_states')(dense_model)

    model = Model(inputs, output)

    tb_callback = TensorBoard(log_dir=log_path + '/', histogram_freq=0, write_graph=True,
                              write_images=True)
    bm_callback = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                  mode='auto')

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', 'binary_crossentropy'])
    return [model, [bm_callback, tb_callback]]