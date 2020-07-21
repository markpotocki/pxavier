import datafile
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from tensorflow.keras import losses


maxFeatures = 1000
sequenceLength = 250

trainingData, testData = datafile.process("/Users/markpotocki/Documents/Workspaces/Resources/ClamAV/daily")

def standardizeData(input):
    pass

vectorizeLayer = TextVectorization(
    standardize=standardizeData,
    max_tokens=maxFeatures,
    output_mode="int",
    output_sequence_length=sequenceLength
)

trainingLabels = [1]*len(testData)
testingLabels = [1]*len(trainingData)
trainingDataset = tf.data.Dataset.from_tensor_slices( (trainingData, [1]*len(trainingData)) )
testDataset = tf.data.Dataset.from_tensor_slices( (testData, trainingLabels) )

# vectorize the data
trainingHashes = trainingDataset.map(lambda x, y: x)
#print(trainingHashes)
#vectorizeLayer.adapt(trainingHashes)

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

trainingDataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
testDataset.batch(BATCH_SIZE)

embeddingDim = 16

"""
model = tf.keras.Sequential([
    layers.Embedding(maxFeatures + 1, embeddingDim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
])
"""

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# model.summary()
model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer="adam", metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
epochs = 10
history = model.fit(
    trainingDataset,
    validation_data=testDataset,
    epochs=epochs
)