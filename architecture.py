from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import tensorflow.keras.backend as K


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class ARCHITECTURE():
    def __init__(self):
        pass
    
    def protEmb(self, input_shape):
        input_target = Input(shape=(input_shape,))
        dense_1 = Dense(256, activation = 'relu',kernel_initializer='glorot_normal')(concat)
        dense_1_dropout = Dropout(0.5)(dense_1)
        dense_2 = Dense(128, activation = 'relu',kernel_initializer='glorot_normal')(dense_1_dropout)
        dense_2_dropout = Dropout(0.2)(dense_2) 
        return dense_2_dropout

    def drugEmb(self, input_shape):
        input_drug = Input(shape=(input_shape,))
        dense_1 = Dense(256, activation = 'relu',kernel_initializer='glorot_normal')(concat)
        dense_1_dropout = Dropout(0.5)(dense_1)
        dense_2 = Dense(128, activation = 'relu',kernel_initializer='glorot_normal')(dense_1_dropout)
        dense_2_dropout = Dropout(0.2)(dense_2) 
        return dense_2_dropout

    def drugDes(self, input_shape):
        input_drug_des = Input(shape=(input_shape,))
        dense_drug_des_1 = Dense(256, activation="relu", kernel_initializer='glorot_normal')(input_drug_des)
        dense_drug_des_2 = Dense(256, activation="relu", kernel_initializer='glorot_normal')(dense_drug_des_1)
        return input_drug_des, dense_drug_des_2

    def protDes(self, input_shape):
        pass
    
    def drugXAE(self, input_shape):
        pass