#include "Sequential.h"

using namespace NeuralNetwork::Model;
using namespace NeuralNetwork::Layers;
using namespace NeuralNetwork::Optimizer;

Sequential(Dense layers[]) : IModel(layers)
{

}

~Sequential()
{
}

bool compile(char lossErrorFunc[], IOptimizer optimizer)
{
}

/*
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
model.predict([100.0])
*/