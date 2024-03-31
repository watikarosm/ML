# Import the libraries we need for this lab

# Using the following line code to install the torchvision library
# !conda install -y torchvision

# PyTorch Library
import torch 
# PyTorch Neural Network
import torch.nn as nn
# Allows us to transform tensors
import torchvision.transforms as transforms
# Allows us to download datasets
import torchvision.datasets as dsets
# Allows us to use activation functions
import torch.nn.functional as F
# Used to graph data and loss curves
import matplotlib.pylab as plt
# Allows us to use arrays to manipulate and store data
import numpy as np
# Setting the seed will allow us to control randomness and give us reproducibility
torch.manual_seed(2)

# Create the model class using Relu as the activation function

class Relu(nn.Module):
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        # D_in is the input size of the first layer (size of input layer)
        # H1 is the output size of the first layer and input size of the second layer (size of first hidden layer)
        # H2 is the outpout size of the second layer and the input size of the third layer (size of second hidden layer)
        # D_out is the output size of the third layer (size of output layer)
        super(Relu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    # Prediction
    def forward(self, x):
        # Puts x through the first layers then the relu function
        x = torch.relu(self.linear1(x))  
        # Puts results of previous line through second layer then relu function
        x = torch.relu(self.linear2(x))
        # Puts result of previous line through third layer
        x = self.linear3(x)
        return x

# Create the model class using Sigmoid as the activation function
class Sigmoid(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        # D_in is the input size of the first layer (size of input layer)
        # H1 is the output size of the first layer and input size of the second layer (size of first hidden layer)
        # H2 is the outpout size of the second layer and the input size of the third layer (size of second hidden layer)
        # D_out is the output size of the third layer (size of output layer)
        super(Sigmoid, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    # Prediction
    def forward(self,x):
        # Puts x through the first layers then the sigmoid function
        x = torch.sigmoid(self.linear1(x)) 
        # Puts results of previous line through second layer then sigmoid function
        x = torch.sigmoid(self.linear2(x))
        # Puts result of previous line through third layer
        x = self.linear3(x)
        return x
    
# Model Training Function
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}  
    # Number of times we train on the entire training dataset
    for epoch in range(epochs):
        # For each batch in the train loader
        for i, (x, y) in enumerate(train_loader):
            # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            optimizer.zero_grad()
            # Makes a prediction on the image tensor by flattening it to a 1 by 28*28 tensor
            z = model(x.view(-1, 28 * 28))
            # Calculate the loss between the prediction and actual class
            loss = criterion(z, y)
            # Calculates the gradient value with respect to each weight and bias
            loss.backward()
            # Updates the weight and bias according to calculated gradient value
            optimizer.step()
            # Saves the loss
            useful_stuff['training_loss'].append(loss.data.item())
        # Counter to keep track of correct predictions
        correct = 0
        # For each batch in the validation dataset
        for x, y in validation_loader:
            # Make a prediction
            z = model(x.view(-1, 28 * 28))
            # Get the class that has the maximum value
            _, label = torch.max(z, 1)
            # Check if our prediction matches the actual class
            correct += (label == y).sum().item()
        # Saves the percent accuracy
        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    return useful_stuff
''''''''''''''''''''''''''' Load data here '''''''''''''''''''''
# Create the training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Create the validating dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
''''''''''''''''''''''''''' End load data '''''''''''''''''''''''
# Create the criterion function
criterion = nn.CrossEntropyLoss()

# Create the training data loader and validation data loader object
# Batch size is 2000 and shuffle=True means the data will be shuffled at every epoch
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
# Batch size is 5000 and the data will not be shuffled at every epoch
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)
# Set the parameters to create the model
input_dim = 28 * 28 # Diemension of an image
hidden_dim1 = 50 # Number of nodes in layer 1
hidden_dim2 = 50 # Number of nodes in layer 2
output_dim = 10 # Number of classes
cust_epochs = 10 # number of iterations

def trainedByReLU(input_dim, hidden_dim1, hidden_dim2, output_dim, lr):
    # Train the model with relu function
    learning_rate = lr
    # Create an instance of the NetRelu model
    modelRelu = Relu(input_dim, hidden_dim1, hidden_dim2, output_dim)
    # Create an optimizer that updates model parameters using the learning rate and gradient
    optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)
    # Train the model
    training_results_relu = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)
    return training_results_relu

def trainedBySig(input_dim, hidden_dim1, hidden_dim2, output_dim, lr):
    # Create the criterion function
    criterion = nn.CrossEntropyLoss()
    # Train the model with sigmoid function
    learning_rate = lr
    # Create an instance of the Net model
    model = Sigmoid(input_dim, hidden_dim1, hidden_dim2, output_dim)
    # Create an optimizer that updates model parameters using the learning rate and gradient
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Train the model
    training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)
    return training_results

# Use case
# Create the training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Create the validating dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Create the training data loader and validation data loader object
# Batch size is 2000 and shuffle=True means the data will be shuffled at every epoch
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
# Batch size is 5000 and the data will not be shuffled at every epoch
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Set the parameters to create the model
input_dim = 28 * 28 # Diemension of an image
hidden_dim1 = 50 # Number of nodes in layer 1
hidden_dim2 = 50 # Number of nodes in layer 2
output_dim = 10 # Number of classes
cust_epochs = 10 # number of iterations
lr = 0.01

training_results = trainedByReLU(input_dim, hidden_dim1, hidden_dim2, output_dim, lr)

training_results_relu = trainedBySig(input_dim, hidden_dim1, hidden_dim2, output_dim, lr)

# Compare the training loss
plt.plot(training_results['training_loss'], label='sigmoid loss')
plt.plot(training_results_relu['training_loss'], label='relu loss')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()
plt.show()
# # Compare the validation loss
plt.plot(training_results['validation_accuracy'], label = 'sigmoid accuracy')
plt.plot(training_results_relu['validation_accuracy'], label = 'relu accuracy') 
plt.ylabel('validation accuracy')
plt.xlabel('Iteration')   
plt.title('training accurary iterations')
plt.legend()
plt.show()