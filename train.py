import argparse
import os
import numpy as np
from PIL import Image
import csv
from numpy import genfromtxt

def one_hot_encode(Y):
    unique_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Create an identity matrix with shape (num_labels, num_labels)
    identity_matrix = np.eye(len(unique_labels))

    # Convert the strings to integer labels
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_dict[label] for label in Y[:, 1]])

    # Create a new 2D array with the encoded labels
    one_hot_encoded_data = np.hstack((Y[:, 0].reshape(-1, 1), identity_matrix[encoded_labels]))
    return one_hot_encoded_data

# for parsing directory type input
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sq(y_true, y_pred):
    return np.mean(((y_true).astype(float) - y_pred)**2)

def ce(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true.astype(float) * np.log((y_pred)+ epsilon))

def initialize_parameters(layer_dims):
    parameters = {}
    num_layers = len(layer_dims)

    for l in range(1, num_layers):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l-1], layer_dims[l])
        parameters[f"b{l}"] = np.zeros(layer_dims[l])

    return parameters

def forward_propagation(X, parameters, activation):
    cache = {"A0": X}
    num_layers = len(parameters) // 2

    if activation == 'sigmoid':
        for l in range(1, num_layers):
            cache[f"Z{l}"] = np.dot(cache[f"A{l-1}"], parameters[f"W{l}"]) + parameters[f"b{l}"]
            cache[f"A{l}"] = sigmoid(cache[f"Z{l}"])

        cache[f"Z{num_layers}"] = np.dot(cache[f"A{num_layers-1}"], parameters[f"W{num_layers}"]) + parameters[f"b{num_layers}"]
        cache[f"A{num_layers}"] = sigmoid(cache[f"Z{num_layers}"])

    else:
        for l in range(1, num_layers):
            cache[f"Z{l}"] = np.dot(cache[f"A{l-1}"], parameters[f"W{l}"]) + parameters[f"b{l}"]
            cache[f"A{l}"] = tanh(cache[f"Z{l}"])

        cache[f"Z{num_layers}"] = np.dot(cache[f"A{num_layers-1}"], parameters[f"W{num_layers}"]) + parameters[f"b{num_layers}"]
        cache[f"A{num_layers}"] = tanh(cache[f"Z{num_layers}"])
    
    return cache

def backward_propagation(X, y, parameters, cache):
    gradients = {}
    num_samples = X.shape[0]
    num_layers = len(parameters) // 2

    dZ = cache[f"A{num_layers}"] - y.astype(float)
    gradients[f"dW{num_layers}"] = np.dot(cache[f"A{num_layers-1}"].T, dZ) / num_samples
    gradients[f"db{num_layers}"] = np.mean(dZ, axis=0)

    for l in range(num_layers-1, 0, -1):
        dA = np.dot(dZ, parameters[f"W{l+1}"].T)
        dZ = dA * cache[f"A{l}"] * (1 - cache[f"A{l}"])
        gradients[f"dW{l}"] = np.dot(cache[f"A{l-1}"].T, dZ) / num_samples
        gradients[f"db{l}"] = np.mean(dZ, axis=0)

    return gradients

def update_parameters(parameters, gradients, learning_rate, optimizer, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
    num_layers = len(parameters) // 2

    if optimizer == "gd":
        for l in range(1, num_layers+1):
            parameters[f"W{l}"] -= learning_rate * gradients[f"dW{l}"]
            parameters[f"b{l}"] -= learning_rate * gradients[f"db{l}"]

    elif optimizer == "momentum":
        for l in range(1, num_layers+1):
            v_dW = beta * gradients[f"dW{l}"] + (1 - beta) * gradients[f"dW{l}"]
            v_db = beta * gradients[f"db{l}"] + (1 - beta) * gradients[f"db{l}"]
            parameters[f"W{l}"] -= learning_rate * v_dW
            parameters[f"b{l}"] -= learning_rate * v_db

    elif optimizer == "adam":
        t = 1
        for l in range(1, num_layers+1):
            v_dW = np.zeros_like(gradients[f"dW{l}"])
            v_db = np.zeros_like(gradients[f"db{l}"])
            s_dW = np.zeros_like(gradients[f"dW{l}"])
            s_db = np.zeros_like(gradients[f"db{l}"])

            v_dW = beta1 * v_dW + (1 - beta1) * gradients[f"dW{l}"]
            v_db = beta1 * v_db + (1 - beta1) * gradients[f"db{l}"]
            s_dW = beta2 * s_dW + (1 - beta2) * gradients[f"dW{l}"]**2
            s_db = beta2 * s_db + (1 - beta2) * gradients[f"db{l}"]**2

            v_dW_corrected = v_dW / (1 - beta1**t)
            v_db_corrected = v_db / (1 - beta1**t)
            s_dW_corrected = s_dW / (1 - beta2**t)
            s_db_corrected = s_db / (1 - beta2**t)

            parameters[f"W{l}"] -= learning_rate * v_dW_corrected / (np.sqrt(s_dW_corrected) + epsilon)
            parameters[f"b{l}"] -= learning_rate * v_db_corrected / (np.sqrt(s_db_corrected) + epsilon)

            t += 1

    return parameters

def mini_batch_generator(X, y, batch_size):
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size

    indices = np.random.permutation(num_samples)

    for b in range(num_batches):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_indices = indices[start:end]

        yield X[batch_indices], y[batch_indices]

    if num_samples % batch_size != 0:
        start = num_batches * batch_size
        batch_indices = indices[start:]

        yield X[batch_indices], y[batch_indices]




def train_model(X, y, layer_dims, learning_rate, num_iterations, batch_size, validation_split, anneal, optimizer, loss, activation, expt_dir):
    num_samples = X.shape[0]
    num_validation_samples = int(num_samples * validation_split)
    num_train_samples = num_samples - num_validation_samples

    # Shuffle and split the data into training and validation sets
    indices = np.random.permutation(num_samples)
    X_shuffled, y_shuffled = X[indices], y[indices]
    X_train, y_train = X_shuffled[:num_train_samples], y_shuffled[:num_train_samples]
    X_val, y_val = X_shuffled[num_train_samples:], y_shuffled[num_train_samples:]

    parameters = initialize_parameters(layer_dims)

    best_val_loss = float("inf")
    restart_epoch = False

    # Create log files
    log_train_path = os.path.join(expt_dir, "log_train.txt")
    log_val_path = os.path.join(expt_dir, "log_val.txt")

    log_train = open(log_train_path, "w")
    log_val = open(log_val_path, "w")

    for i in range(num_iterations):
        if restart_epoch:
            restart_epoch = False
            continue

        for X_batch, y_batch in mini_batch_generator(X_train, y_train, batch_size):
            cache = forward_propagation(X_batch, parameters, activation)
            gradients = backward_propagation(X_batch, y_batch, parameters, cache)
            parameters = update_parameters(parameters, gradients, learning_rate, optimizer)

        if (i + 1) % 100 == 0:
            train_cache = forward_propagation(X_train, parameters, activation)

            if loss =='ce':
                train_loss = ce(y_train, train_cache[f"A{len(layer_dims)-1}"])
                val_cache = forward_propagation(X_val, parameters, activation)
                val_loss = ce(y_val, val_cache[f"A{len(layer_dims)-1}"])
                print(f"Iteration {i+1}: Train Loss = {train_loss}, Validation Loss = {val_loss}")


            else:
                train_loss = sq(y_train, train_cache[f"A{len(layer_dims)-1}"])
                val_cache = forward_propagation(X_val, parameters, activation)
                val_loss = sq(y_val, val_cache[f"A{len(layer_dims)-1}"])
                print(f"Iteration {i+1}: Train Loss = {train_loss}, Validation Loss = {val_loss}")

            train_predictions = np.argmax(train_cache[f"A{len(layer_dims)-1}"], axis=1)
            train_labels = np.argmax(y_train, axis=1)
            train_error_rate = 100 * np.mean(train_predictions != train_labels)

            val_predictions = np.argmax(val_cache[f"A{len(layer_dims)-1}"], axis=1)
            val_labels = np.argmax(y_val, axis=1)
            val_error_rate = 100 * np.mean(val_predictions != val_labels)

            # Write loss and error rate to log files
            log_train.write(f"STEPS {i+1}: Train Loss = {train_loss}, Error Rate = {train_error_rate}, lr = {learning_rate}\n\n")
            log_val.write(f"STEPS {i+1}: Validation Loss = {val_loss}, Error Rate = {val_error_rate}, lr = {learning_rate}\n\n")


            if anneal and val_loss < best_val_loss:
                best_val_loss = val_loss 
            else:
                learning_rate /= 2
                restart_epoch = True

    return parameters




parser = argparse.ArgumentParser(description='taking parameters')

parser.add_argument('--lr', type = float, metavar='', required=True, help = 'learning_rate')
parser.add_argument('--momentum', type = float, metavar='', required=True, help = 'momentum')
parser.add_argument('--num_hidden', type = int, metavar='', required=True, help = 'no_of_hiddenlayers')
parser.add_argument('--activation', type = str, metavar='', required=True)
parser.add_argument('--loss', type = str, metavar='', required=True)
parser.add_argument('--batch_size', type = int, metavar='', required=True)
parser.add_argument('--opt', type = str, metavar='', required=True)
parser.add_argument('--anneal', type = bool, metavar='', required=True)
parser.add_argument('--sizes', type = str, metavar='', required=True)
parser.add_argument('--save_dir', type=dir_path, metavar='', required = True)
parser.add_argument('--expt_dir', type=dir_path, metavar='', required = True)
parser.add_argument('--train', type=dir_path, metavar='', required = True)
parser.add_argument('--test', type=dir_path, metavar='', required = True)


args = parser.parse_args()
d = vars(parser.parse_args()) # this is a dict

if "sizes" in d.keys():
    d["sizes"] = [int(s.strip()) for s in d["sizes"].split(",")]

size_of_layers = d["sizes"]
lr = args.lr
momentum = args.momentum
num_hidden = args.num_hidden
activation = args.activation
loss = args.loss
batch_size = args.batch_size
opt = args.opt
anneal = args.anneal
save_dir = args.save_dir
expt_dir = args.expt_dir
train_loc = args.train
test_loc = args.test

sizes = [1024]
for i in size_of_layers:
    sizes.append(i)

sizes.append(10)

epochs = 200
validation_split = 0.2


#creating dataset out of given images
label_path = 'trainlabels.csv'
data = np.genfromtxt(label_path, delimiter=',', dtype='object', skip_header=1)
column1 = data[:, 0].astype(int)
column2 = data[:, 1].astype(str)

# Create a NumPy array from the extracted columns
x_labels = np.column_stack((column1, column2))
x_labels= one_hot_encode(x_labels)
x_labels = np.delete(x_labels, 0, axis = 1)


train_path = train_loc
def image_to_row(image_path):
    image = Image.open(image_path).convert("L")
    image_array = np.array(image)
    row = image_array.flatten()
    return row

#Iterate through the png files in the folder and convert each image into a row
matrix = []
for filename in os.listdir(train_path):
    if filename.endswith(".png"):
        image_path = os.path.join(train_path, filename)
        row = image_to_row(image_path)
        matrix.append(row)

# Convert the list of rows into a 2D matrix
matrix = np.array(matrix)

# Print the shape of the resulting matrix
x_train = matrix/255.0

print(matrix.shape)

parameters = train_model(x_train, x_labels, sizes, lr, epochs, batch_size, validation_split, anneal, opt, loss, activation, expt_dir)

file_path = save_dir + '/' + 'parameters.txt'

with open(file_path, 'w') as file:
    file.write("Directory Location: " + save_dir + "\n\n")
    file.write("Dictionary:\n")
    for key, value in parameters.items():
        file.write(key + ":\n")
        file.write(str(value) + "\n\n")

#testing 
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

test_path = test_loc
matrix2 = []
for filename in os.listdir(test_path):
    if filename.endswith(".png"):
        image_path = os.path.join(test_path, filename)
        row = image_to_row(image_path)
        matrix2.append(row)

matrix2 = np.array(matrix2)

# Print the shape of the resulting matrix
x_test = matrix2/255.0

all_weights = forward_propagation(x_test, parameters, activation)
prediction = all_weights[f"A{len(sizes)-1}"]

# decoder
max_indices = np.argmax(prediction, axis=1)
decoded_labels = [labels[index] for index in max_indices]
decoded_matrix = np.column_stack((np.arange(x_test.shape[0]), decoded_labels))
print(decoded_matrix.shape)


directory = expt_dir
filename = "test_submission.csv"

with open(directory + '/' + filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "label"])  # Write header row
    for row_idx, row in enumerate(decoded_matrix):
        key, value = row
        writer.writerow([key, value])

