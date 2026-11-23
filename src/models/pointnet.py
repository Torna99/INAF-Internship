import os
os.environ["KERAS_BACKEND"] = "tensorflow"  
import keras
from keras import layers, ops

@keras.saving.register_keras_serializable(package="tgf")
class MaskedMax(layers.Layer):
    """
        Max over the hits axis, ignoring padded hits.
    """
    def call(self, x, valid):
        v = ops.expand_dims(valid, -1)                  # (batch, hits, 1)
        return ops.max(x + (v - 1.0) * 1e9, axis=1)     # padded hits wont be the max

def shared_mlp(x, sizes):
    """
        Same Dense weights applied to every hit.
    """
    for s in sizes:
        x = layers.Dense(s, activation="relu")(x)
    return x

def build_pointnet(n_classes, max_hits=64, n_hit=8, n_glob_feat=2, per_hit=(64, 128, 256), head=(128, 64), use_local_global=False):
    hits  = keras.Input((max_hits, n_hit), name="hits")
    valid = keras.Input((max_hits,), name="valid")
    glob  = keras.Input((n_glob_feat,), name="glob")

    # Feature extraction 
    local = shared_mlp(hits, per_hit) 

    # symmetric aggregation: max over hits -> one global vector per event of dim (batch, 256)
    global_hit_per_event = MaskedMax()(local, valid)                  

    # OPTIONAL: give each hit the global context and process it again (as in PointNet segmentation), then aggregate again
    if use_local_global:
        g_rep = layers.RepeatVector(max_hits)(global_hit_per_event)             # (batch, hits, 256)
        mixed = layers.Concatenate()([local, g_rep])                            # (batch, hits, 512)
        global_hit_per_event = MaskedMax()(shared_mlp(mixed, (128, 256)), valid)

    # classifier head, with the two event-level numbers (n_hits, E_tot)
    x = layers.Concatenate()([global_hit_per_event, glob])
    for s in head:
        x = layers.Dense(s, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model([hits, valid, glob], out)
