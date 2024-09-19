import numpy as np

class conv4x4:

    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 4, 4) / 16

    
    def generate_regions(self, board):
        height, width = 6, 7
        for i in range(height - 3):
            for j in range(width - 3):
                region = board[i:(i + 4), j:(j + 4)]
                yield region, i, j
    
    def forward(self, board):
        height, width = board.shape
        output = np.zeros((height - 3, width - 3, self.num_filters))
        for region, i, j in self.generate_regions(board):
            output[i, j] = np.sum(region * self.filters, axis = (1, 2))
        return output
    
test = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 2, 0, 0, 0],
    [1, 0, 1, 2, 0, 0, 2]
])

conv = conv4x4(8)
output = conv.forward(test)
print(output.shape)







#CHATGPT OUTPUT:

# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return (x > 0).astype(float)

# def convolve4x4(image, kernel, stride=1, padding=0):
#     # Add padding to the input image
#     image = np.pad(image, [(padding, padding), (padding, padding)], mode='constant', constant_values=0)
#     kernel_height, kernel_width = kernel.shape
#     image_height, image_width = image.shape
#     output_height = (image_height - kernel_height) // stride + 1
#     output_width = (image_width - kernel_width) // stride + 1
#     output = np.zeros((output_height, output_width))

#     for y in range(0, output_height):
#         for x in range(0, output_width):
#             output[y, x] = np.sum(image[y*stride:y*stride+kernel_height, x*stride:x*stride+kernel_width] * kernel)
    
#     return output

# def max_pooling2d(image, size=2, stride=2):
#     image_height, image_width = image.shape
#     output_height = (image_height - size) // stride + 1
#     output_width = (image_width - size) // stride + 1
#     output = np.zeros((output_height, output_width))

#     for y in range(0, output_height):
#         for x in range(0, output_width):
#             output[y, x] = np.max(image[y*stride:y*stride+size, x*stride:x*stride+size])
    
#     return output

# class SimpleCNN:
#     def __init__(self):
#         # Initialize convolutional kernel
#         self.kernel = np.random.randn(3, 3)  # 3x3 kernel
#         self.fc_weights = np.random.randn(12*12, 10)  # Fully connected weights (example size)
#         self.fc_bias = np.random.randn(10)

#     def forward(self, x):
#         # Convolution
#         self.conv_out = convolve2d(x, self.kernel, stride=1, padding=1)
#         self.relu_out = relu(self.conv_out)
        
#         # Max Pooling
#         self.pool_out = max_pooling2d(self.relu_out, size=2, stride=2)
        
#         # Flatten
#         self.pool_out_flat = self.pool_out.flatten()
        
#         # Fully Connected Layer
#         self.fc_out = np.dot(self.pool_out_flat, self.fc_weights) + self.fc_bias
#         self.fc_out = relu(self.fc_out)
        
#         return self.fc_out
    
#     # Create a simple 8x8 image with random values
# image = np.random.randn(8, 8)

# # Initialize and run the CNN
# cnn = SimpleCNN()
# output = cnn.forward(image)

# print("Output from CNN:", output)