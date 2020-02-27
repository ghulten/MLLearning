# Hello world level PyTorch neural network with GPU


#####
# **** Dataset processing will create child precesses, so use this idiom to avoid rerunning everything for each child 
#####
if __name__ == '__main__':

   ####
   # **** Create the dataset objects and training generator
   ####

   import DataHelpers

   (xTrainFilePaths, yTrain, xTestFilePaths, yTest) = DataHelpers.LoadRawData('dataset_B_Eye_Images')

   import EyeDataset
   trainDataSet = EyeDataset.EyeDataset(xTrainFilePaths, yTrain)
   testDataSet = EyeDataset.EyeDataset(xTestFilePaths, yTest)

   from torch.utils import data

   batchSize = 1024
   trainDataSetGenerator = data.DataLoader(trainDataSet, batch_size=batchSize, shuffle=True, num_workers=0)
   testDataSetGenerator = data.DataLoader(testDataSet, batch_size=batchSize, shuffle=True, num_workers=0)
      
   ####
   # Set up for the training run
   ####

   import torch

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   print("Device is:", device)

   import FullyConnectedNetwork
   model = FullyConnectedNetwork.FullyConnectedNetwork(hiddenNodes=5)

   model = model.to(device)

   lossFunction = torch.nn.BCELoss()

   optimizer = torch.optim.SGD(model.parameters(), lr=1.5e-3)


   ####
   # **** Train the model
   ####

   import time
   startTrainingTime = time.time()

   for i in range(2500): # run for this many epochs
      
      optimizer.zero_grad()
      
      for batchXTensor, batchYTensor in trainDataSetGenerator:
         batchXTensorGPU = batchXTensor.to(device)
         batchYTensorGPU = batchYTensor.to(device)
         
         yTrainPredicted = model(batchXTensorGPU)
         
         loss = lossFunction(yTrainPredicted, batchYTensorGPU)
         
         loss.backward()
         
      optimizer.step()
      
      print("Epoch: %d" % (i))

   endTrainingTime = time.time()
   print("Training complete. Time: %s" % (endTrainingTime - startTrainingTime))

   ####
   # Test the model
   ####

   model.train(mode=False)

   numCorrect = 0
   numTested = 0
   for batchXTensor, batchYTensor in testDataSetGenerator:
      batchXTensorGPU = batchXTensor.to(device)

      yTestPredicted = model(batchXTensorGPU)

      predictions = [ 1 if probability > .5 else 0 for probability in yTestPredicted ]

      correct = [ 1 if predictions[i] == batchYTensor[i] else 0 for i in range( len(predictions) ) ]
      
      numCorrect += sum(correct)
      numTested += len(correct)

   print("Accuracy: %.2f" % ( numCorrect / numTested ) )