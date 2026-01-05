"""
Model classes for training - handle dict inputs
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class BMF(keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50):
        super(BMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = layers.Embedding(n_users, embedding_dim)
        self.item_embedding = layers.Embedding(n_items, embedding_dim)

        # User and item biases
        self.user_bias = layers.Embedding(n_users, 1)
        self.item_bias = layers.Embedding(n_items, 1)

        # Global bias
        self.global_bias = self.add_weight(
            shape=(1,), initializer='zeros', trainable=True, name='global_bias'
        )

    def call(self, inputs, training=None):
        # Handle dict inputs for training
        if isinstance(inputs, dict):
            user_ids = inputs['user_ids']
            item_ids = inputs['item_ids']
        else:
            # Handle tuple inputs for inference
            user_ids, item_ids = inputs

        # Get embeddings and biases
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        user_b = self.user_bias(user_ids)
        item_b = self.item_bias(item_ids)

        # Compute dot product
        dot_product = tf.reduce_sum(user_emb * item_emb, axis=-1, keepdims=True)

        # Final prediction with biases
        output = dot_product + user_b + item_b + self.global_bias

        return output


class NeuMF(keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_dims=[64, 32, 16]):
        super(NeuMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # GMF (Generalized Matrix Factorization) embeddings
        self.user_embedding_gmf = layers.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = layers.Embedding(n_items, embedding_dim)

        # MLP (Multi-Layer Perceptron) embeddings
        self.user_embedding_mlp = layers.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = layers.Embedding(n_items, embedding_dim)

        # MLP layers
        self.mlp_layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            self.mlp_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.mlp_layers.append(layers.Dropout(0.2))
            input_dim = hidden_dim

        # Final prediction layer (combines GMF and MLP)
        self.final_layer = layers.Dense(1)

    def call(self, inputs, training=None):
        # Handle dict inputs for training
        if isinstance(inputs, dict):
            user_ids = inputs['user_ids']
            item_ids = inputs['item_ids']
        else:
            # Handle tuple inputs for inference
            user_ids, item_ids = inputs

        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = tf.multiply(user_emb_gmf, item_emb_gmf)

        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = tf.concat([user_emb_mlp, item_emb_mlp], axis=-1)

        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output, training=training)

        # Combine GMF and MLP
        combined = tf.concat([gmf_output, mlp_output], axis=-1)

        # Final prediction
        output = self.final_layer(combined)

        return output


class LNCM(keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_dims=[64, 32]):
        super(LNCM, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = layers.Embedding(n_users, embedding_dim)
        self.item_embedding = layers.Embedding(n_items, embedding_dim)

        # Linear combination layer
        self.linear_layer = layers.Dense(1, use_bias=True)

        # Neural layers
        self.hidden_layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            self.hidden_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.hidden_layers.append(layers.Dropout(0.2))
            input_dim = hidden_dim

        # Neural output layer
        self.neural_layer = layers.Dense(1)

        # Combination weight
        self.alpha = self.add_weight(
            shape=(1,), initializer='uniform', trainable=True, name='alpha'
        )

    def call(self, inputs, training=None):
        # Handle dict inputs for training
        if isinstance(inputs, dict):
            user_ids = inputs['user_ids']
            item_ids = inputs['item_ids']
        else:
            # Handle tuple inputs for inference
            user_ids, item_ids = inputs

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Linear part (Matrix Factorization)
        linear_output = self.linear_layer(tf.concat([user_emb, item_emb], axis=-1))

        # Neural part
        neural_input = tf.concat([user_emb, item_emb], axis=-1)
        neural_output = neural_input
        for layer in self.hidden_layers:
            neural_output = layer(neural_output, training=training)
        neural_output = self.neural_layer(neural_output)

        # Combine linear and neural parts
        output = tf.sigmoid(self.alpha) * linear_output + (1 - tf.sigmoid(self.alpha)) * neural_output

        return output


class ENCM(keras.Model):
    def __init__(self, n_users, n_items, n_contexts, embedding_dim=50, context_dims=None, hidden_dims=[64, 32]):
        super(ENCM, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # Use provided context_dims or default to uniform dimension
        if context_dims is None:
            context_dims = [10] * len(n_contexts)
        elif len(context_dims) != len(n_contexts):
            raise ValueError("context_dims must have same length as n_contexts")

        # User and item embeddings
        self.user_embedding = layers.Embedding(n_users, embedding_dim)
        self.item_embedding = layers.Embedding(n_items, embedding_dim)

        # Context embeddings with variable dimensions
        self.context_embeddings = []
        total_context_dim = 0
        for i, (n_context, context_dim) in enumerate(zip(n_contexts, context_dims)):
            self.context_embeddings.append(
                layers.Embedding(n_context, context_dim, name=f'context_{i}')
            )
            total_context_dim += context_dim

        # Neural layers for processing interactions
        self.hidden_layers = []
        input_dim = embedding_dim * 2 + total_context_dim

        for hidden_dim in hidden_dims:
            self.hidden_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.hidden_layers.append(layers.Dropout(0.2))
            input_dim = hidden_dim

        # Output layer
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=None):
        # Handle dict inputs for training
        if isinstance(inputs, dict):
            user_ids = inputs['user_ids']
            item_ids = inputs['item_ids']
            context_features = inputs['context_features']
        else:
            # Handle tuple inputs for inference
            user_ids, item_ids, context_features = inputs

        # Get user and item embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Get context embeddings
        context_embs = []
        for i, context_embedding in enumerate(self.context_embeddings):
            context_embs.append(context_embedding(context_features[:, i]))

        # Concatenate all features
        if context_embs:
            all_features = tf.concat([user_emb, item_emb] + context_embs, axis=-1)
        else:
            all_features = tf.concat([user_emb, item_emb], axis=-1)

        # Pass through neural network
        output = all_features
        for layer in self.hidden_layers:
            output = layer(output, training=training)

        # Final prediction
        prediction = self.output_layer(output)

        return prediction
