from tensorflow.keras.layers import Conv2D, Input 
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
import tensorflow as tf


def SRCNN915():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=9, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=32, kernel_size=1, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=5, padding='valid',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN915")

def SRCNN935():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=9, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=5, padding='valid',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)

    return Model(X_in, X_out, name="SRCNN935")

def SRCNN955():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=9, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=32, kernel_size=5, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=5, padding='valid',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)

    return Model(X_in, X_out, name="SRCNN955")
