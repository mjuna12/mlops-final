import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

# Define constants
LABEL_KEY_NEW = "sentiment"
FEATURE_KEY_NEW = "review"
EMBEDDING_DIM = 16

# Function to rename transformed features


def transformed_name(key):
    """
    Transform the given key.

    Args:
        key (str): Input key to transform.

    Returns:
        str: Transformed key.
    """
    return key + "_xf"

# Function to read data from compressed TFRecord files


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Input function to create transformed features and batch data


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    """
    Create input function for training data.

    Args:
        file_pattern (str): File pattern for input data.
        tf_transform_output (tensorflow_transform.TFTransformOutput): TensorFlow Transform output.
        num_epochs (int): Number of epochs.
        batch_size (int): Batch size.

    Returns:
        tf.data.Dataset: Input dataset.
    """
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY_NEW)
    )
    return dataset


# Text vectorization layer for tokenization and data standardization
vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=100
)

# Function to build the machine learning model


def model_builder():
    """
    Build the machine learning model.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    FEATURE_KEY = FEATURE_KEY_NEW  # Using FEATURE_KEY_NEW value to define FEATURE_KEY

    inputs = tf.keras.Input(
        shape=(
            1,
        ),
        name=transformed_name(FEATURE_KEY),
        dtype=tf.string)
    # Wrap tf.reshape in a Lambda layer
    reshaped_input = layers.Lambda(lambda x: tf.reshape(x, [-1]))(inputs)
    x = vectorize_layer(reshaped_input)
    x = layers.Embedding(10000, EMBEDDING_DIM, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Get serving function for TensorFlow Serving."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        # Remove the sentiment key from feature_spec to let TFT handle it
        feature_spec.pop(LABEL_KEY_NEW)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        # TFT will handle the sentiment transformation, no need to do it here.
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch')

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        patience=10)

    # Modify the checkpoint path to end with .keras
    checkpoint_path = os.path.join(
        fn_args.serving_model_dir,
        'checkpoints',
        'best_model.keras')
    mc = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,  # Ensure the path ends with .keras
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Use only the feature (not the label) when adapting the vectorizer
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    # Adapt the vectorizer layer with only the transformed feature
    vectorize_layer.adapt(
        train_set.map(lambda x, y: x[transformed_name(FEATURE_KEY_NEW)])
    )

    model = model_builder()

    steps_per_epoch = len(list(train_set))
    validation_steps = len(list(val_set))

    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, es, mc],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=10
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model,
            tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))}
    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures=signatures)
