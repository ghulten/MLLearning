# Hello world level PyTorch neural network with GPU


####
# Load the data
####

import DataHelpers
from PIL import Image

(xTrainFilePaths, yTrain, xTestFilePaths, yTest) = DataHelpers.LoadRawData('dataset_B_Eye_Images')

xTrainImages = [ Image.open(path) for path in xTrainFilePaths ]
xTestImages = [ Image.open(path) for path in xTestFilePaths ]

####
# Get the data into PyTorch's data structures
####

import torch
from torchvision import transforms

xTrainTensor = torch.stack( [ transforms.ToTensor()(image) for image in xTrainImages ] )
yTrainTensor = torch.Tensor( [ [ yValue ] for yValue in yTrain ] )

xTestTensor = torch.stack( [ transforms.ToTensor()(image) for image in xTestImages ] )

#### 
# Move the data onto the GPU
####

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is:", device)

xTrainTensorGPU = xTrainTensor.to(device)
yTrainTensorGPU = yTrainTensor.to(device)

xTestTensorGPU = xTestTensor.to(device)


####
# Set up for the training run
####

import FullyConnectedNetwork
model = FullyConnectedNetwork.FullyConnectedNetwork(hiddenNodes=5)

model = model.to(device)

lossFunction = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1.5e-3)


####
# Train the model
####

import time
startTrainingTime = time.time()

for i in range(2500):
   yTrainPredicted = model(xTrainTensorGPU)
   
   loss = lossFunction(yTrainPredicted, yTrainTensorGPU)
   
   optimizer.zero_grad()
   
   loss.backward()
   
   optimizer.step()
   
   print("Iteration: %d loss: %.4f" % (i, loss.item()))

endTrainingTime = time.time()
print("Training complete. Time: %s" % (endTrainingTime - startTrainingTime))

####
# Test the model
####

model.train(mode=False)

yTestPredicted = model(xTestTensorGPU)

predictions = [ 1 if probability > .5 else 0 for probability in yTestPredicted ]

correct = [ 1 if predictions[i] == yTest[i] else 0 for i in range( len(predictions) ) ]

print("Accuracy: %.2f" % ( sum(correct) / len(correct) ) )