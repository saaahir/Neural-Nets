#include "Matrix.h" 
#include "Network.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    int numTrainingData = 100;
    int numTestingData = 1;
    int numClasses = 10;
    int numHiddenLayers = 2;
    int layerSize[4] = {28*28, 16, 16, 10};
    double learningRate = 3;
    int epochs = 10;
    int miniBatchSize = 1;
    // char* trainingfile = "ex.csv";
    char* testingfile = "b.csv";
    char* trainingfile = "mnist_train.csv";
    // char* testingfile = "mnist_test.csv";
    char* biasfile = "biases.bin";
    char* weightfile = "weights.bin";

    Example** train = createExamplesFromFile(numTrainingData, numClasses, trainingfile); 
    Example** test = createExamplesFromFile(numTestingData, numClasses, testingfile); 
    Network *network = createNetwork(train, numHiddenLayers, layerSize, learningRate, epochs, miniBatchSize);

    loadWeights(network, weightfile);
    loadBiases(network, biasfile);

    double accuracy = evaluate(test, network, numTestingData);
    printf("Accuracy: %f\n", accuracy); //

    // SGD(network, numTrainingData);

    // accuracy = evaluate(test, network, numTestingData);
    // printf("Accuracy: %f\n", accuracy); //

    // saveWeights(network, weightfile);
    // saveBiases(network, biasfile);

    destroyExamples(train, numTrainingData); 
    destroyExamples(test, numTestingData); 
    destroyNetwork(network);    

}
