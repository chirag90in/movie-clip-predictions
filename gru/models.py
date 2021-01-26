import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score

from functools import partial


'''
classification
'''

def GRUClassifier(X, k_layers=3, k_hidden=16, 
                  l2=0.001, dropout=1e-6, lr=0.006, seed=42):
    
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    input_layers = [layers.Masking(mask_value=0.0, 
                                   input_shape = [None, X.shape[-1]])]
    
    hidden_layers = []
    for ii in range(k_layers):
        hidden_layers.append(CustomGRU(k_hidden,return_sequences=True))
        
    output_layer = [layers.TimeDistributed(layers.Dense(15,activation='softmax'))]
    
    optimizer = keras.optimizers.Adam(lr=lr)
    
    model = keras.models.Sequential(input_layers+hidden_layers+output_layer)
    
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    
    return model


def GRUEncoder(X, l2=0.001, dropout=1e-6, lr=0.006,seed=42):
    
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    model = keras.models.Sequential([layers.Masking(mask_value=0.0, 
                                                             input_shape=[None, X.shape[-1]]),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     layers.TimeDistributed(layers.Dense(3,activation='linear')),
                                     layers.TimeDistributed(layers.Dense(15,activation='softmax'))
                                    ])

    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    
    return model


'''
classifier (FeedForward)
k_feat: (k_hidden:)*k_layers: k_class
'''
def FFClassifier(X,k_hidden,k_layers):
    input_layers = [layers.Masking(mask_value=0.0, input_shape = [X.shape[-2], X.shape[-1]])]
    FF_layers = []
    for ii in range(k_layers):
        if ii == 0:
            FF_layers.append(layers.Dense(k_hidden))
        elif ii == k_layers-1:
            FF_layers.append(layers.Dense(15,activation='softmax'))
        else:
            FF_layers.append(layers.Dense(k_hidden))

    model = keras.models.Sequential(input_layers+FF_layers)
    
    optimizer = keras.optimizers.Adam()
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    return model

'''
regression
'''

def GRURegressor(X,k_layers=3, k_hidden=16, 
                 l2=0, dropout=0, lr=0.001,seed=42):
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    input_layers = [layers.Masking(mask_value=0.0, 
                                   input_shape = [None, X.shape[-1]])]
    
    hidden_layers = []
    for ii in range(k_layers):
        hidden_layers.append(CustomGRU(k_hidden,return_sequences=True))
        
    output_layer = [layers.TimeDistributed(layers.Dense(1,activation='linear'))]
    
    optimizer = keras.optimizers.Adam(lr=lr)
    
    model = keras.models.Sequential(input_layers+hidden_layers+output_layer)
    
    model.compile(loss='mse',
                      optimizer=optimizer)
    
    return model


########################## OLD ########################################
def GRUClassifierOLD(X, y, l2=0.001, dropout=1e-6, lr=0.006, 
               epochs=20, batch_size=32,seed=42):
    
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    model = keras.models.Sequential([layers.Masking(mask_value=0.0, 
                                                             input_shape=[None, X.shape[-1]]),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     layers.TimeDistributed(layers.Dense(15,activation='softmax'))
                                    ])

    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    model.fit(X,y,epochs=epochs,
                  validation_split=0.2,batch_size=batch_size,verbose=1)
    return model


def GRURegressorOLD(l2=0, dropout=0, lr=0.001,seed=42):
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    model = keras.models.Sequential([layers.Masking(mask_value=0.0, 
                                                             input_shape=[None, 300]),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     layers.TimeDistributed(layers.Dense(1,activation='linear'))
                                    ])
    # Optimizer
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='mse',
                      optimizer=optimizer)
    return model