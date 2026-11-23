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


@keras.saving.register_keras_serializable(package="tgf")
class Adjacency(layers.Layer):
    """
        Connect two valid hits if their distance is <= radius (normalised units).
    """
    def __init__(self, radius=0.40, **kw):
        """
            To effectevily choose the radius sqrt(3) / VOXEL SIZEE
        """      
        super().__init__(**kw)
        self.radius = radius

    def call(self, pos, valid):                 
        diff = pos[:, :, None, :] - pos[:, None, :, :]

        dist = ops.sqrt(ops.sum(diff ** 2, axis=-1) + 1e-9)        # Pitagora 3D

        adj = ops.cast(dist < self.radius, "float32")              # 1 if two different hit are close (distance <= radius)
        adj = adj * valid[:, :, None] * valid[:, None, :]          # remove padded hits

        return adj - ops.eye(adj.shape[1])[None]                   # no self-loops (remove the 1 in diagonals which are self-loops)

    # save the radius value in exporting the model
    def get_config(self):
        return {**super().get_config(), "radius": self.radius}

@keras.saving.register_keras_serializable(package="tgf")
class GraphConv(layers.Layer):
    """
        One message-passing step: new = relu(W_self * x + W_nb * mean(neighbours)).
    """
    def __init__(self, units, **kw):
        super().__init__(**kw)
        self.units = units
        self.self_dense = layers.Dense(units)
        self.nb_dense = layers.Dense(units)

    def call(self, x, adj):                       
        """
            Returns:  
                x: (b, hits, f), 
                adj: (b, hits, hits)
        """
        deg = ops.sum(adj, axis=-1, keepdims=True) + 1e-6       # Counter the number of neigh for each hit (degree of the node)
        neighbours = ops.matmul(adj, x) / deg                   # Isolated hits get zeros
        return ops.relu(self.self_dense(x) + self.nb_dense(neighbours))
    
    def get_config(self):
        return {**super().get_config(), "units": self.units}


def build_gnn(n_classes, max_hits=64, n_hit=8, n_glob_feat=2, dim_embedding=64, n_layers=3, radius=0.40):
    hits  = keras.Input((max_hits, n_hit), name="hits")
    valid = keras.Input((max_hits,), name="valid")
    glob  = keras.Input((n_glob_feat,), name="glob")

    pos = hits[:, :, :3]                           # absolute voxel position (the 3 coords already normalized)
    adj = Adjacency(radius)(pos, valid)            # Compute the Adjacency Matrix: graph built inside the model

    ## initial embedding (each input from the original n_hit feature to vector of number of dimension dim_embedding)
    x = layers.Dense(dim_embedding, activation="relu")(hits)  

    ## Each hit knows the characteristic of their n_layer-hop neighbors
    for _ in range(n_layers):                      # each layer sees one hop further
        x = GraphConv(dim_embedding)(x, adj)

    ## Pooling and classification
    g = MaskedPool()(x, valid)                     # (b, 2*d + 1): mean, max, log1p(n)
    z = layers.Concatenate()([g, glob])
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(64, activation="relu")(z)
    out = layers.Dense(n_classes, activation="softmax")(z)
    return keras.Model([hits, valid, glob], out)



