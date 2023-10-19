import numpy as np

def tf_pooling(A_prev, hparameters, mode="average"):
    """
    Optimized pooling layer implementation using numpy operations.
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    # Retrieve dimensions from the input shape and hparameters
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    # Create a matrix to hold the values we need for the pooling operation.
    # Then, using stride, select the appropriate elements from A_prev and compute the operation.
    for h in range(n_H):
        for w in range(n_W):
            h_start = h * stride
            h_end = h_start + f
            w_start = w * stride
            w_end = w_start + f
            
            # Extract a slice of A_prev
            A_prev_slice = A_prev[:, h_start:h_end, w_start:w_end, :]
            
            # Perform the pooling operation
            if mode == "max":
                A[:, h, w, :] = np.max(A_prev_slice, axis=(1, 2))
            elif mode == "average":
                A[:, h, w, :] = np.mean(A_prev_slice, axis=(1, 2))

    # Verify that the output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))

    # Store the input and hparameters in "cache" for the pooling_backward function
    cache = (A_prev, hparameters)
    
    return A, cache

# The rest of the code would be testing this function with appropriate input.
