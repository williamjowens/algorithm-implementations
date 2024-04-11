import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, d_k):
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = np.matmul(attention_weights, V)
    return output, attention_weights, scores

def multi_head_attention(Q, K, V, d_model, num_heads):
    batch_size, sequence_length, _ = Q.shape
    d_k = d_model // num_heads

    Q_reshaped = Q.reshape(batch_size, sequence_length, num_heads, d_k)
    K_reshaped = K.reshape(batch_size, sequence_length, num_heads, d_k)
    V_reshaped = V.reshape(batch_size, sequence_length, num_heads, d_k)

    Q_transposed = Q_reshaped.transpose(0, 2, 1, 3)
    K_transposed = K_reshaped.transpose(0, 2, 1, 3)
    V_transposed = V_reshaped.transpose(0, 2, 1, 3)

    attention_output, attention_weights, attention_scores = scaled_dot_product_attention(Q_transposed, K_transposed, V_transposed, d_k)

    attention_output_reshaped = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
    attention_weights_reshaped = attention_weights.transpose(0, 2, 3, 1).reshape(batch_size, sequence_length, sequence_length, num_heads)
    attention_scores_reshaped = attention_scores.transpose(0, 2, 3, 1).reshape(batch_size, sequence_length, sequence_length, num_heads)

    return attention_output_reshaped, attention_weights_reshaped, attention_scores_reshaped

def visualize_matrix(matrix, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    # Example input data
    batch_size = 1
    sequence_length = 5
    d_model = 64
    num_heads = 8

    # Generate random query, key, and value matrices
    Q = np.random.randn(batch_size, sequence_length, d_model)
    K = np.random.randn(batch_size, sequence_length, d_model)
    V = np.random.randn(batch_size, sequence_length, d_model)

    # Perform multi-head attention
    attention_output, attention_weights, attention_scores = multi_head_attention(Q, K, V, d_model, num_heads)

    print("Input sequence length:", sequence_length)
    print("d_model:", d_model)
    print("Number of attention heads:", num_heads)
    print("Attention output shape:", attention_output.shape)
    print("Attention weights shape:", attention_weights.shape)
    print("Attention scores shape:", attention_scores.shape)

    # Visualize matrices
    visualize_matrix(Q[0], "Query Matrix")
    visualize_matrix(K[0], "Key Matrix")
    visualize_matrix(V[0], "Value Matrix")
    visualize_matrix(attention_scores[0, :, :, 0], "Attention Scores (Head 0)")
    visualize_matrix(attention_weights[0, :, :, 0], "Attention Weights (Head 0)")
    visualize_matrix(attention_output[0], "Attention Output")

if __name__ == "__main__":
    main()