import torch

def torch_pooling(A_prev, hparameters, mode="max"):
    """
    Custom pooling function implemented in PyTorch.

    Arguments:
    A_prev -- Input data, torch Tensor of shape (m, n_C_prev, n_H_prev, n_W_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a torch Tensor of shape (m, n_C, n_H, n_W)
    """

    # Retrieve dimensions from the input shape and hparameters
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev  # Number of channels remains unchanged

    # Initialize output tensor A
    A = torch.zeros((m, n_C, n_H, n_W))

    # Loop over the vertical, horizontal axis, and depth of the output volume
    # Use the corners to define the current slice from the input tensor A_prev
    for h in range(n_H):
        for w in range(n_W):
            h_start = h * stride
            h_end = h_start + f
            w_start = w * stride
            w_end = w_start + f

            # Extract a slice of A_prev
            A_prev_slice = A_prev[:, :, h_start:h_end, w_start:w_end]

            # Perform the pooling operation (max or average) on the extracted slice
            # and store the result in the output tensor A at the correct position
            if mode == "max":
                A[:, :, h, w] = torch.max(A_prev_slice, dim=2).values.max(dim=2).values
            elif mode == "average":
                A[:, :, h, w] = torch.mean(A_prev_slice, dim=(2, 3))

    # Making sure your output shape is correct
    assert(A.shape == (m, n_C, n_H, n_W))

    return A

# Test and debug the function with a dummy tensor
if __name__ == "__main__":
    torch.manual_seed(1)  # for reproducibility
    A_prev_torch = torch.randn(2, 3, 5, 5)  # creating a random tensor
    hparameters = {"f": 3, "stride": 1}

    # Test the function
    A_torch = torch_pooling(A_prev_torch, hparameters, mode="max")
    print("Max pooling result: \n", A_torch)
    A_torch = torch_pooling(A_prev_torch, hparameters, mode="average")
    print("Average pooling result: \n", A_torch)
