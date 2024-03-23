import numpy as np

# Activation function
def sigmoid_fn(x):
    result = 1 / (1 + np.exp(-x))
    return result

# Function derivative
def sigmoid_derivative(x):
    result = x * (1 - x)
    return result

# Initialize network parameters
def initialize_parameters(input_neurons, hidden_neurons, output_neurons):
    weights_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    bias_hidden = np.random.uniform(size=(1, hidden_neurons))
    weights_output = np.random.uniform(size=(hidden_neurons, output_neurons))
    bias_output = np.random.uniform(size=(1, output_neurons))
    
    return weights_hidden, bias_hidden, weights_output, bias_output

# Forward propagation
def forward_propagation(X, weights_hidden, bias_hidden, weights_output, bias_output):
    hidden_layer_activation = np.dot(X, weights_hidden) + bias_hidden
    hidden_layer_output = sigmoid_fn(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, weights_output) + bias_output
    predicted_output = sigmoid_fn(output_layer_activation)
    
    return hidden_layer_output, predicted_output

# Backward propagation
def backward_propagation(X, y, hidden_layer_output, predicted_output, weights_output):
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    return d_hidden_layer, d_predicted_output, error 

# Update parameters
def update_parameters(X, hidden_layer_output, d_hidden_layer, d_predicted_output, 
                      weights_hidden, bias_hidden, weights_output, bias_output, lr
):
    weights_output += hidden_layer_output.T.dot(d_predicted_output) * lr
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    weights_hidden += X.T.dot(d_hidden_layer) * lr
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr
    
    return weights_hidden, bias_hidden, weights_output, bias_output

# Training function
def train_network(X, y, input_neurons, hidden_neurons, output_neurons, epochs, lr):
    weights_hidden, bias_hidden, weights_output, bias_output = initialize_parameters(
        input_neurons, hidden_neurons, output_neurons
    )
    
    for epoch in range(epochs):
        hidden_layer_output, predicted_output = forward_propagation(
            X, weights_hidden, bias_hidden, weights_output, bias_output
        )
        d_hidden_layer, d_predicted_output, error = backward_propagation(
            X, y, hidden_layer_output, predicted_output, weights_output
        )
        weights_hidden, bias_hidden, weights_output, bias_output = update_parameters(
            X, hidden_layer_output, d_hidden_layer, d_predicted_output,
            weights_hidden, bias_hidden, weights_output, bias_output, lr
        )
        
        if epoch % 1000 == 0:
            print(f"Error at epoch {epoch} is {np.mean(np.abs(error))}")
            
    return weights_hidden, bias_hidden, weights_output, bias_output, predicted_output

# Classify raw output
def classify_output(output, threshold=0.5):
    result = (output > threshold).astype(int)
    return result


# Main block
if __name__ == "__main__":
    # Input and output data for XOR with 2 inputs
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Training the network
    input_neurons = 2
    hidden_neurons = 2
    output_neurons = 1
    epochs = 10000
    lr = 0.1
    
    final_weights_hidden, final_bias_hidden, \
    final_weights_output, final_bias_output, predicted_output = train_network(
        X, y, input_neurons, hidden_neurons, output_neurons, epochs, lr
    )
    
    # Print the final predicted output
    print(f"Final predicted output:\n{predicted_output}")
    
    # Classify output
    classified_output = classify_output(predicted_output)
    print(f"Final classification decision:\n{classified_output}")