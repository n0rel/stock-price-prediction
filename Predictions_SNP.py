from datetime import time
from numpy.random.mtrand import uniform
import pandas as pd
import numpy as np
from numpy.random import uniform
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
from datetime import datetime
import time

class LSTMCell:
    def __init__(self, features):
        self.hidden = 0
        self.previous_output = 0

        # Weights. Each row represents a different gate, ex: 0 -> forget, 1 -> input, etc...
        self.hdn_w = np.random.rand(4, features) * math.sqrt(1/(features+1))
        self.inp_w = np.random.rand(4, features) * math.sqrt(1/(features + 1))


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forget_gate(self, x, prev_y) -> tuple:
        linc = sum(x * self.inp_w[0] + prev_y * self.hdn_w[0])
        gate_output = self.sigmoid(linc)
        return gate_output, linc

    def input_gate(self, x, prev_y):
        linc = sum(x * self.inp_w[1] + prev_y * self.hdn_w[1])
        gate_output = self.sigmoid(linc)

        linc2 = sum(x * self.inp_w[2] + prev_y * self.hdn_w[2])
        return gate_output, linc, linc2

    def output_gate(self, x, prev_y):
        linc = sum(x * self.inp_w[3] + prev_y * self.hdn_w[3])
        gate_output = self.sigmoid(linc)
        return gate_output, linc

    def predict(self, x):
        for time_step in range(len(x)):

            # Forget Gate
            forget_gate, _ = self.forget_gate(x[time_step], self.previous_output)
            self.hidden *= forget_gate

            # Input Gate
            input_gate, lci1, lci2 = self.input_gate(x[time_step], self.previous_output)
            self.hidden += input_gate * np.tanh(lci2)

            # Output Gate
            output_gate, _ = self.output_gate(x[time_step], self.previous_output)

            self.previous_output = np.tanh(self.hidden) * output_gate

        return self.previous_output


class Model:
    def __init__(self, features, cell_count):
        self.features = features
        self.cell_count = cell_count

        # Create the cells & weights and initalize them
        self.cells = [LSTMCell(self.features) for _ in range(self.cell_count)]
        self.weights = np.random.rand(cell_count) * math.sqrt(1/(cell_count))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def predict(self, x):
        # Pass through our input to each cell and collect the predictions
        cell_predictions = []
        for cell in self.cells:
            cell_predictions.append(cell.predict(x))

        # Combine the predictions with the model weights
        linear_combination = sum(np.array(cell_predictions).T * self.weights)
        return linear_combination

    def fit(self, x, y, epochs, learning_rate):
        """
        Using the input information and cell functions, predicts a value 1 timestep in the future by
        fitting the input dataset on the cell using backpropagation through time (BPTT)
        :param x: Numpy Array of size (timesteps, features) representing the input
        :return: A number representing a prediction
        """

        for epoch in range(epochs):
            print("Starting Epoch " + str(epoch))
            time_start = time.time()

            for batch in x:
                
                hidden_states = [[] for _ in range(self.cell_count)]
                linc_f = [[] for _ in range(self.cell_count)]
                linc_i1 = [[] for _ in range(self.cell_count)]
                linc_i2 = [[] for _ in range(self.cell_count)]
                linc_o = [[] for _ in range(self.cell_count)]
                input_gates = [[] for _ in range(self.cell_count)]
                hidden_before_forget = [[] for _ in range(self.cell_count)]
                cell_outputs = [[] for _ in range(self.cell_count)]
                overall_outputs = []
                linc_overall = []

                # Forward Pass
                for time_step in range(len(batch)):
                    
                    for num, cell in enumerate(self.cells):
                        # Forget Gate
                        forget_gate, lcf = cell.forget_gate(batch[time_step], cell.previous_output)
                        cell.hidden *= forget_gate

                        hidden_before_forget[num].append(cell.hidden)

                        # Input Gate
                        input_gate, lci1, lci2 = cell.input_gate(batch[time_step], cell.previous_output)
                        cell.hidden += input_gate * np.tanh(lci2)

                        # Output Gate
                        output_gate, lco = cell.output_gate(batch[time_step], cell.previous_output)
                        output = np.tanh(cell.hidden) * output_gate
                        cell.previous_output = output
                        cell_outputs[num].append(output)

                        linc_f[num].append(lcf), linc_i1[num].append(lci1), linc_i2[num].append(lci2), linc_o[num].append(lco)
                        input_gates[num].append(input_gate)
                        hidden_states[num].append(cell.hidden)

                    # Combine each cell output into a linear combination and sigmoid it
                    linc_output = sum([cell_output[-1] * self.weights[num] for num,cell_output in enumerate(cell_outputs)])
                    linc_overall.append(linc_output)
                    overall_outputs.append(self.sigmoid(linc_output))

                # Back propagate for each time step
                hdn_gradients = np.zeros(shape=(self.cell_count, 4, self.features))
                inp_gradients = np.zeros(shape=(self.cell_count, 4, self.features))
                model_gradients = np.zeros(shape=self.cell_count)

                for time_step in range(1, len(batch)+1):
                    overall_outputs = np.array(overall_outputs)

                    # Updating Model Weights
                    d_model = (overall_outputs[-time_step] - y[-time_step]) * (self.sigmoid_derivative(linc_overall[-time_step]))
                    model_gradients += d_model * np.array([z[-time_step] for z in cell_outputs])

                    for num,cell in enumerate(self.cells):
                        # Updating Output Weights
                        d_output = d_model * (self.weights[num]) * (np.tanh(hidden_states[num][-time_step])) * (
                                self.sigmoid_derivative(linc_o[num][-time_step]))
                        hdn_gradients[num] += d_output * hidden_states[num][-time_step]
                        inp_gradients[num] += d_output * batch[-time_step]


                        # Updating Input2 Weights
                        d_input2 = d_model * (self.weights[num]) * (cell_outputs[num][-time_step]) * (
                                    1 - np.power(np.tanh(hidden_states[num][-time_step]), 2)) * (
                                    self.sigmoid_derivative(hidden_states[num][-time_step])) * (input_gates[num][-time_step]) * (
                                    1 - np.power(np.tanh(linc_i2[num][-time_step]), 2))

                        hdn_gradients[num] += d_input2 * cell_outputs[num][-time_step]
                        inp_gradients[num] += d_input2 * batch[-time_step]

                        # Updating Input1 Weights
                        d_input1 = d_model * (self.weights[num]) * (cell_outputs[num][-time_step]) * (
                                    1 - np.power(np.tanh(hidden_states[num][-time_step]), 2)) * (
                                    self.sigmoid_derivative(hidden_states[num][-time_step])) * (np.tanh(linc_i2[num][-time_step])) * (
                                    self.sigmoid_derivative(linc_i1[num][-time_step]))

                        hdn_gradients[num] += d_input1 * cell_outputs[num][-time_step]
                        inp_gradients[num] += d_input1 * batch[-time_step]

                        # Updating Forget Weights
                        d_forget = d_model * (self.weights[num]) * (cell_outputs[num][-time_step]) * (
                                    1 - np.power(np.tanh(hidden_states[num][-time_step]), 2)) * (
                                    self.sigmoid_derivative(hidden_states[num][-time_step])) * (hidden_before_forget[num][-time_step]) * (
                                    self.sigmoid_derivative(linc_f[num][-time_step]))

                    hdn_gradients[num] += d_forget * cell_outputs[num][-time_step]
                    inp_gradients[num] += d_forget * batch[-time_step]

            self.weights -= learning_rate * (model_gradients / len(batch))
            for num, cell in enumerate(self.cells):
                cell.hdn_w -= learning_rate * (hdn_gradients[num] / len(batch))
                cell.inp_w -= learning_rate * (hdn_gradients[num] / len(batch))

            print(f"- Time took: {time.time() - time_start}")

def get_data(data_length) -> pd.DataFrame:
    # Import the CSV's ('Path', 'Delimiter', [Columns to use])
    paths_info = [('SNP500.csv', ',', [3]), ]
    datasets = [np.flip(np.genfromtxt(f"Data\{path[0]}", delimiter=path[1], usecols=path[2])) for path in paths_info]

    # Add Bias
    datasets = np.insert(datasets, len(datasets), np.zeros(data_length,), axis=0)
    data = np.stack(datasets, axis=-1)

    return data


def split_train_test(data, timesteps, length, train_length) -> tuple:
    training = []
    testing = []
    for i in range(timesteps, train_length):
        training.append([data[i-timesteps:i], data[i, 0]])

    for i in range(train_length, length):
        testing.append([data[i-timesteps:i], data[i, 0]])

    return np.array(training, dtype=object), np.array(testing, dtype=object)

# Prepare Data
price_scaler = MinMaxScaler(feature_range=(0, 1))
data = get_data(data_length=2473)


data[:, 0] = price_scaler.fit_transform(np.reshape(data[:, 0], (-1, 1)))[:, 0]


# Training, Validation, & Test Sets
features = 2
timesteps = 60
learning_rate = 0.001
epochs = 1
cell_count = 1
training, testing = split_train_test(data, timesteps=timesteps, length=len(data), train_length=2000)

# Create model and predict
time_start = time.time()

model = Model(features=features, cell_count=cell_count)
model.fit(training[:, 0], training[:, 1], learning_rate=learning_rate, epochs=epochs)

print(f"Time Taken: {time.time()-time_start}s")


predictions = []
real = []
error = 0
for x, y in zip(testing[:, 0], testing[:, 1]):
    prediction = model.predict(x)
    scaled = price_scaler.inverse_transform(prediction.reshape(1, -1))[0]
    scaled_real = price_scaler.inverse_transform(y.reshape(1, -1))[0]
    predictions.append(scaled)
    real.append(scaled_real)
    error += price_scaler.inverse_transform(y.reshape(1, -1))[0] - scaled

# Print and plot the predictions
print(f"ERROR: {error / len(testing)}")
plt.plot(predictions)
plt.plot(real)
plt.legend(["Prediction", "Output"])
plt.text(s="Learning Rate: " + str(learning_rate) + " Epochs: " + str(epochs), x=-7.2, y=200)
plt.show()
