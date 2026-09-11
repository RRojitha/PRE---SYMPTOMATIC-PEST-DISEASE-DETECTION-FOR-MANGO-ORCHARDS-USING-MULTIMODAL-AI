import tensorflow as tf

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal_and_vertical"),

    tf.keras.layers.RandomRotation(0.15),

    tf.keras.layers.RandomZoom(0.10),

    tf.keras.layers.RandomContrast(0.10),

], name="augmentation")