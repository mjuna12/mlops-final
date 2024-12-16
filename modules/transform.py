
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY_NEW = "sentiment"
FEATURE_KEY_NEW = "review"


def transformed_name(key):
    """Append '_xf' to the feature key to denote it as transformed."""
    return key + "_xf"


def preprocessing_fn(inputs):
    """Preprocesses the input features.

    Args:
        inputs: A dictionary of input features.

    Returns:
        A dictionary of transformed features.
    """
    outputs = {}

    # Ensure the required keys are in the input data
    if FEATURE_KEY_NEW not in inputs:
        raise KeyError(
            f"Feature key '{FEATURE_KEY_NEW}' is missing from the input data.")
    if LABEL_KEY_NEW not in inputs:
        raise KeyError(
            f"Label key '{LABEL_KEY_NEW}' is missing from the input data.")

    # Preprocess the review feature (convert to lowercase and strip whitespace)
    review_feature = inputs[FEATURE_KEY_NEW]
    review_feature = tf.strings.lower(review_feature)  # Convert to lowercase
    review_feature = tf.strings.strip(review_feature)  # Remove extra spaces
    # Save transformed review feature
    outputs[transformed_name(FEATURE_KEY_NEW)] = review_feature

    # Transform the sentiment label (string to numeric mapping)
    sentiment_feature = inputs[LABEL_KEY_NEW]
    sentiment_feature = tf.strings.lower(sentiment_feature)
    sentiment_feature = tf.where(
        tf.equal(sentiment_feature, "positive"),
        tf.constant(1.0, dtype=tf.float32),
        tf.constant(0.0, dtype=tf.float32)  # Map "negative" to 0.0
    )
    outputs[transformed_name(LABEL_KEY_NEW)] = sentiment_feature

    # Print output untuk debugging
    print(f"Outputs from preprocessing_fn: {outputs}")

    return outputs
