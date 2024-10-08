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
    
    def forward(self, input):
        self.last_input = input
        height, width = input.shape
        output = np.zeros((height - 3, width - 3, self.num_filters))
        for region, i, j in self.generate_regions(input):
            output[i, j] = np.sum(region * self.filters, axis = (1, 2))
        return output
    
    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        self.filters -= learn_rate * d_L_d_filters

        return None
    
class MaxPool:

    prev_input = []

    def generate_regions(self, board):
        height, width, _ = board.shape
        new_height = height // 2
        new_width = width // 2
        for i in range(new_height):
            for j in range(new_width):
                region = board[(i * 2):(i* 2 + 2), (j * 2):(j * 2 + 2)]
                yield region, i ,j
    
    def forward(self, input):
        self.prev_input = input
        height, width, num_filters = input.shape
        output = np.zeros((height // 2, width // 2, num_filters))
        for region, i, j in self.generate_regions(input):
            output[i, j] = np.amax(region, axis = (0, 1))
        return output
    
    def backprop(self, d_loss_d_out):
        d_loss_d_input = np.zeros(self.prev_input.shape)

        for im_region, i, j in self.generate_regions(self.prev_input):
            height, width, depth = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(height):
                for j2 in range(width):
                    for f2 in range(depth):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_loss_d_input[i * 2 + i2, j * 2 + j2, f2] = d_loss_d_out[i, j, f2]

        return d_loss_d_input
    
class MaxPool2:

    prev_input = []

    def forward(self, input):
        self.prev_input = input



    
    
class Softmax:

    prev_input_shape = 0
    prev_input = []
    prev_totals = []

    def __init__(self, input_len, nodes):
        self.weights = np.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.prev_input_shape = input.shape
        input = input.flatten()
        self.prev_input = input
        input_len, nodes = self.weights.shape
        totals = np.dot(input, self.weights) + self.biases
        self.prev_totals = totals
        exp = np.exp(totals)
        return exp/np.sum(exp, axis = 0)

    def backprop(self, d_loss_d_out, learn_rate):
        for i, gradient in enumerate(d_loss_d_out):
            if gradient == 0:
                continue
            totals_exp = np.exp(self.prev_totals)      
            S = np.sum(totals_exp)
            d_out_d_total = -totals_exp[i] * totals_exp / (S ** 2)
            d_out_d_total[i] = totals_exp[i] * (S - totals_exp[i]) / (S ** 2)
            # Gradients of totals against weights/biases/input
            d_total_d_weight = self.prev_input
            d_total_d_bias = 1
            d_total_d_inputs = self.weights
            # Gradients of loss against totals
            d_loss_d_total = gradient * d_out_d_total
            # Gradients of loss against weights/biases/input
            d_loss_d_weight = d_total_d_weight[np.newaxis].T @ d_loss_d_total[np.newaxis]
            d_loss_d_bias = d_loss_d_total * d_total_d_bias
            d_loss_d_inputs = d_total_d_inputs @ d_loss_d_total

            # Update weights / biases
            self.weights -= learn_rate * d_loss_d_weight
            self.biases -= learn_rate * d_loss_d_bias
            return d_loss_d_inputs.reshape(self.prev_input_shape)
        
def train(im, label, lr=.005, ):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # Forward
    out, loss, acc = Softmax.forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = Softmax.backprop(gradient, lr)
    # TODO: backprop MaxPool2 layer
    # TODO: backprop Conv3x3 layer

    return loss, acc
    
    
test = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 2, 0, 0, 0],
    [1, 0, 1, 2, 0, 0, 2]
])

pool = MaxPool()
conv = conv4x4(8)
output = conv.forward(test)
output = pool.forward(output)
print(output.shape)

print('MNIST CNN initialized!')

train_images = None
train_labels = None

# Train!
loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
        print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0

l, acc = train(im, label)
loss += l
num_correct += acc






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