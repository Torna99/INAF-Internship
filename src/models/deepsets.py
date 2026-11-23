import os
os.environ["KERAS_BACKEND"] = "tensorflow"  
import keras
from keras import layers, ops

@keras.saving.register_keras_serializable(package="tgf")
class MaskedPool(layers.Layer):
    """
        Mean and Max on event's hits ignoring the padding.
    """
    def call(self, x, valid):
        v = ops.expand_dims(valid, -1) # Mask with 0 and 1 for padding or real hit
        n_real_event = ops.sum(v, axis=1)
        mean = ops.sum(x * v, axis=1) / (n_real_event + 1e-6)
        max = ops.max(x + (v - 1.0) * 1e9, axis=1)      
        return ops.concatenate([mean, max, ops.log1p(n_real_event)], axis=-1) # returns the vector of mean, vector o maxs and the log of the number of hit

    def compute_output_shape(self, x_shape, valid_shape=None):
        return (x_shape[0], 2 * x_shape[-1] + 1)


def build_deepsets(n_classes, feats_train=None, max_hits=64, n_feat=8, n_glob_feat=2, dim_embedding=64):
    """
        Create a deepsets net following the mathematical formula:
            rho(sum phi(x_i))
        
    """
    hits  = keras.Input((max_hits, n_feat), name="hits")
    valid = keras.Input((max_hits,), name="valid")
    glob  = keras.Input((n_glob_feat,), name="glob")
    inputs, extra = [hits, valid, glob], [glob]

    ## Phi: elaborate single hit
    x = layers.Dense(64, activation="relu")(hits)     
    x = layers.Dense(dim_embedding, activation="relu")(x)
    pooled = MaskedPool()(x, valid)                    

    # ## Optional: to use also the computer faetures (42)
    # if feats_train is not None:                        
    #     feats = keras.Input((feats_train.shape[1],), name="feats")
    #     norm = layers.Normalization()
    #     norm.adapt(feats_train)                        
    #     inputs.append(feats)
    #     extra.append(norm(feats))

    z = layers.Concatenate()([pooled] + extra)
    z = layers.Dense(128, activation="relu")(z)        # ρ: classificatore
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(64, activation="relu")(z)
    out = layers.Dense(n_classes, activation="softmax")(z)
    return keras.Model(inputs, out) 