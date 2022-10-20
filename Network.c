#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "Matrix.h"
#include "Network.h" 

/** TODO: Add malloc null checks everywhere (one-liner style) & call destroy network if null*/
Network* createNetwork(Example **exampleList, int numHiddenLayers, int *layerSizes, double learningRate, int epochs, int miniBatchSize){
    Network *network = malloc(sizeof(Network));
    Matrix *input = exampleList[0]->input;
    network->input = unroll(input);
    network->exampleList = exampleList;
    network->numHiddenLayers = numHiddenLayers;
    network->layerSizes = malloc(sizeof(double) * (numHiddenLayers + 2)); // Includes both input layer and output layer
    network->learningRate = learningRate;
    network->epochs = epochs;
    network->miniBatchSize = miniBatchSize;

    // network->layerSizes[0] = network->input->rows;
    for (int i = 0; i < numHiddenLayers + 2; i++) {
        network->layerSizes[i] = layerSizes[i];
    }


    // network->outputLayer = malloc(sizeof(Matrix) * network->layerSizes[numHiddenLayers]);
    network->hiddenLayers = malloc(sizeof(Matrix *) * network->numHiddenLayers);
    network->biases       = malloc(sizeof(Matrix *) * (network->numHiddenLayers + 1));
    network->weights      = malloc(sizeof(Matrix *) * (network->numHiddenLayers + 1));
    // Do i need this?
    // network->expected     = malloc(sizeof(int) * network->layerSizes[network->numHiddenLayers + 1]);


    // Creating weights
    for (int i = 0; i < numHiddenLayers + 1; i++) {
        Matrix *matrix = createMatrix(network->layerSizes[i], network->layerSizes[i + 1], NULL);
        randomizeMatrix(matrix, -1, 1); /** TODO: WHAT SHOULD THE RANDOMIZED BOUNDS BE?? */
        /** TODO: Prob use gaussian distribution for this */
        network->weights[i] = matrix;
        network->biases[i] = createMatrix(network->layerSizes[i+1], 1, NULL);
        if (i < numHiddenLayers) network->hiddenLayers[i] = NULL;
    }

    return network;
}

void destroyNetwork(Network* network) {
    UNUSED(network);
    // /** TODO: Definitely have to add more stuff to free the matrices here */
    for (int i = 0; i < network->numHiddenLayers + 1; i++) {
        if (i < network->numHiddenLayers) destroyMatrix(network->hiddenLayers[i]);
        destroyMatrix(network->weights[i]);
        destroyMatrix(network->biases[i]);
    }
    destroyMatrix(network->input);
    destroyMatrix(network->outputLayer);


    free(network->hiddenLayers);
    free(network->weights);
    free(network->biases);
    free(network->layerSizes);
    free(network);
}

void destroyExamples(Example** examples, int numData) {
    for(int i = 0; i < numData; i++) {
        destroyMatrix(examples[i]->input);
        free(examples[i]->expected);
        free(examples[i]);
    }

    free(examples);

}

double sigmoid(double one, double two) {
    UNUSED(two);
    return (1 / (1 + exp(-one)));
}

double sigmoidPrime(double one, double two) {
    return sigmoid(one, two) * (1 - sigmoid(one, two));
}

void forward(Network *network) {
    destroyMatrix(network->hiddenLayers[0]);
    Matrix *transposedMatrix = transpose(network->weights[0]);
    network->hiddenLayers[0] = dot(transposedMatrix, network->input);
    destroyMatrix(transposedMatrix);
    add(network->hiddenLayers[0], network->biases[0]);
    applyFunction(network->hiddenLayers[0], sigmoid, 0);
    for (int i = 1; i < network->numHiddenLayers; i++) {
        destroyMatrix(network->hiddenLayers[i]); 
        transposedMatrix = transpose(network->weights[i]);
        network->hiddenLayers[i] = dot(transposedMatrix, network->hiddenLayers[i - 1]);
        add(network->hiddenLayers[i], network->biases[i]);
        applyFunction(network->hiddenLayers[i], sigmoid, 0);
        destroyMatrix(transposedMatrix);
    }

    destroyMatrix(network->outputLayer);
    transposedMatrix = transpose(network->weights[network->numHiddenLayers]);
    network->outputLayer = dot(transposedMatrix, network->hiddenLayers[network->numHiddenLayers - 1]);
    add(network->outputLayer, network->biases[network->numHiddenLayers]);
    applyFunction(network->outputLayer, sigmoid, 0);
    destroyMatrix(transposedMatrix);
}

Matrix** backpropForward(Network *network) {

    Matrix** zList = malloc(sizeof(Matrix*) * (network->numHiddenLayers + 2));


    destroyMatrix(network->hiddenLayers[0]);
    Matrix *transposedMatrix = transpose(network->weights[0]);
    network->hiddenLayers[0] = dot(transposedMatrix, network->input);
    destroyMatrix(transposedMatrix);
    add(network->hiddenLayers[0], network->biases[0]);
    zList[0] = copyMatrix(network->hiddenLayers[0]);
    applyFunction(network->hiddenLayers[0], sigmoid, 0);
    int i = 1;
    for (i = 1; i < network->numHiddenLayers; i++) {
        destroyMatrix(network->hiddenLayers[i]); 
        transposedMatrix = transpose(network->weights[i]);
        network->hiddenLayers[i] = dot(transposedMatrix, network->hiddenLayers[i - 1]);
        add(network->hiddenLayers[i], network->biases[i]);
        zList[i] = copyMatrix(network->hiddenLayers[i]);
        applyFunction(network->hiddenLayers[i], sigmoid, 0);
        destroyMatrix(transposedMatrix);
    }

    destroyMatrix(network->outputLayer);
    transposedMatrix = transpose(network->weights[network->numHiddenLayers]);
    network->outputLayer = dot(transposedMatrix, network->hiddenLayers[network->numHiddenLayers - 1]);
    add(network->outputLayer, network->biases[network->numHiddenLayers]);
    zList[i] = copyMatrix(network->outputLayer);
    applyFunction(network->outputLayer, sigmoid, 0);
    destroyMatrix(transposedMatrix);

    return zList;
}

double loss(Network *network) {
    double total = 0;
    for (int i = 0; i < network->layerSizes[network->numHiddenLayers + 1]; i++) {
        double diff = (network->outputLayer->nums[i][0] - network->expected[i]);
        double squared = diff * diff;
        total += squared;
    }
    double MSE = total / (network->layerSizes[network->numHiddenLayers + 1]); 
    return MSE;
}

Matrix* lossDerivative(Network *network) {
    int numClasses = network->layerSizes[network->numHiddenLayers + 1];
    double *expectedDouble = malloc(sizeof(double) * numClasses);

    for (int i = 0; i < numClasses; i++) {
        expectedDouble[i] = -1.0 * network->expected[i];
    }


    Matrix *expectedMatrix = createMatrix(network->outputLayer->rows, network->outputLayer->cols, expectedDouble);
    free(expectedDouble);

    add(expectedMatrix, network->outputLayer);
    return expectedMatrix;


}

void SGD(Network *network, int numData) {
    int miniBatchSize = network->miniBatchSize;
    int numMiniBatches = numData / miniBatchSize;
    Example*** miniBatches = malloc(sizeof(Example**) * numMiniBatches);
    for (int i = 0; i < network->epochs; i++) {
        shuffle(network->exampleList, numData);
        int offset = 0;
        for(int i = 0; i < numMiniBatches; i++) {
            Example** miniBatch = malloc(sizeof(Example*) * miniBatchSize);
            for(int j = 0; j < miniBatchSize; j++) {
                miniBatch[j] = network->exampleList[j + offset];
            }
            offset += miniBatchSize;
            miniBatches[i] = miniBatch;
        }
        for (int i = 0; i < numMiniBatches; i++) {
            updateMiniBatch(network, miniBatches[i]);
        }
        
        for (int i = 0; i < numMiniBatches; i++) {
            free(miniBatches[i]);
        }

        double accuracy = evaluate(network->exampleList, network, numData);
        printf("Epoch: %d Accuracy: %f\n", i, accuracy); //
        // printf("Epoch: %d\n", i); //

    }
    free(miniBatches);
    /** TODO: I SHOULD DO SOMETHING TO MITIGATE THE PROBLEM THAT OCCURS WHEN THE MINIBATCHSIZE IS NOT A MULTIPLE OF THE NUMDATA*/
}

void updateMiniBatch(Network *network, Example **miniBatch) {
    Matrix **weightGradient = malloc(sizeof(Matrix*) * (network->numHiddenLayers + 1));
    Matrix **biasGradient = malloc(sizeof(Matrix*) * (network->numHiddenLayers+ 1));

    for (int i = 0; i < network->numHiddenLayers + 1; i++) {
        weightGradient[i] = createMatrix(network->layerSizes[i], network->layerSizes[i + 1], NULL);
        biasGradient[i] = createMatrix(network->layerSizes[i+1], 1, NULL);
    }


    for (int i = 0; i < network->miniBatchSize; i++) {
        Gradient *gradient = backprop(network, miniBatch[i]);
        for (int j = 0; j < network->numHiddenLayers + 1; j++) {
            add(weightGradient[j], gradient->delWeights[j]);
            add(biasGradient[j], gradient->delBiases[j]);
            destroyMatrix(gradient->delWeights[j]);
            destroyMatrix(gradient->delBiases[j]);
        }
        free(gradient->delBiases);
        free(gradient->delWeights);
        free(gradient);
        
    }

    for(int i = 0; i < network->numHiddenLayers + 1; i++) {
        scale(weightGradient[i], -(network->learningRate)/network->miniBatchSize);
        scale(biasGradient[i], -(network->learningRate)/network->miniBatchSize);
        add(network->weights[i], weightGradient[i]);
        add(network->biases[i], biasGradient[i]);
        destroyMatrix(weightGradient[i]);
        destroyMatrix(biasGradient[i]);
    }
    free(weightGradient);
    free(biasGradient);
}

Gradient *backprop(Network* network, Example* example) {
    Gradient *gradient = malloc(sizeof(Gradient));
    gradient->delWeights = malloc(sizeof(Matrix*) * (network->numHiddenLayers+ 1));
    gradient->delBiases = malloc(sizeof(Matrix*) * (network->numHiddenLayers+ 1));
    for (int i = 0; i < network->numHiddenLayers + 1; i++) {
        gradient->delWeights[i] = createMatrix(network->layerSizes[i], network->layerSizes[i + 1], NULL);
        gradient->delBiases[i] = createMatrix(network->layerSizes[i+1], 1, NULL);
    }
    destroyMatrix(network->input);
    network->input = unroll(example->input);
    network->expected = example->expected;
    Matrix* lossDeriv = lossDerivative(network);

    Matrix** zList = backpropForward(network);
    
    Matrix* outputLayer = copyMatrix(zList[network->numHiddenLayers]);
    destroyMatrix(zList[network->numHiddenLayers]);
    applyFunction(outputLayer, sigmoidPrime, 0);

    multiply(lossDeriv, outputLayer);
    destroyMatrix(outputLayer);
    destroyMatrix(gradient->delBiases[network->numHiddenLayers]);
    gradient->delBiases[network->numHiddenLayers] = lossDeriv;

    Matrix *transposedActivation = transpose(network->hiddenLayers[network->numHiddenLayers - 1]);
    Matrix *dottedMatrix = dot(lossDeriv, transposedActivation);
    destroyMatrix(gradient->delWeights[network->numHiddenLayers]);
    gradient->delWeights[network->numHiddenLayers] = transpose(dottedMatrix);
    destroyMatrix(transposedActivation);
    destroyMatrix(dottedMatrix);

    for (int i = network->numHiddenLayers - 1; i >= 0; i--) {
        destroyMatrix(gradient->delWeights[i]);
        destroyMatrix(gradient->delBiases[i]);

        outputLayer = copyMatrix(zList[i]);
        destroyMatrix(zList[i]);
        applyFunction(outputLayer, sigmoidPrime, 0);

        // Take advantage of the fact that gradient->delBiases[i + 1] is just equal to delta
        dottedMatrix = dot(network->weights[i+1], gradient->delBiases[i + 1]);
        multiply(dottedMatrix, outputLayer);
        destroyMatrix(outputLayer);

        gradient->delBiases[i] = dottedMatrix;
        Matrix *transposedMatrix;
        if (i == 0)transposedMatrix = transpose(network->input);
        else transposedMatrix = transpose(network->hiddenLayers[i - 1]);

        Matrix* temp = dot(dottedMatrix, transposedMatrix);
        gradient->delWeights[i] = transpose(temp);  

        destroyMatrix(transposedMatrix);
        destroyMatrix(temp);
    }

    free(zList);

    return gradient;
}

void shuffle(Example** exampleList, int numData) {
    srand(time(NULL));
    for (int i = 0; i < numData - 1; i++) {
        int j = rand() % (numData - i) + i;
        Example* temp = exampleList[i];
        exampleList[i] = exampleList[j];
        exampleList[j] = temp;
    }
}

Example** createExamples(int numData, Matrix** inputList, int *labelList, int numClasses) { 
    // Prob gonna have to do some special case when there is only one output neuron
    Example** exampleList = malloc(sizeof(Example*) * numData);
    for (int i = 0; i < numData; i++) {
        Example *example = malloc(sizeof(Example));
        example->expected = malloc(sizeof(int) * numClasses); // sizeof(int) * number of classes (aka the number of nodes in the output layer)
        /** TODO: Maybe unroll this inputlist matrix so i dont have to do it later*/
        // example->input = unroll(inputList[i]);
        example->input = inputList[i];
        for (int j = 0; j < numClasses; j++) {
                example->expected[j] = 0;
        }
        example->expected[labelList[i]] = 1;

        exampleList[i] = example;
    }
    return exampleList;
}

// Specific to MNIST
Example** createExamplesFromFile(int numData, int numClasses, char* filename) { 

    FILE *f = fopen(filename, "r");
    if (!f) return NULL;

    int inputSize = 28 * 28;

    int* labelList = malloc(sizeof(int) * numData);
    Matrix** inputList = malloc(sizeof(Matrix*) * numData);

    double* nums = malloc(sizeof(double) * inputSize);

    int listIdx = 0;
    int numsIdx = 0;


    char *line = NULL;
    size_t len;

    for (int count = 0; count < numData; count++) {
        int num;
        getline(&line, &len, f);

        char *test = strtok(line, ",");
        int label = atoi(test);

        while (test) {
            test = strtok(NULL, ",");
            if (test) num = atoi(test);
            nums[numsIdx] = num/255.0; // THIS IS VERY SPECIFIC TO MNIST, SCALING B/W 0/1
            numsIdx++;
        }

        labelList[listIdx] = label;
        inputList[listIdx] = createMatrix(28, 28, nums);
        numsIdx = 0;
        listIdx++;

    }

    free(line);




    Example** examples = createExamples(numData, inputList, labelList, numClasses);

    free(inputList);
    free(labelList);
    free(nums);
    fclose(f);
    return examples;
}

Result* argmax(Network* network) {
    Result* result = malloc(sizeof(Result));
    int* expected = network->expected;
    double** outputs = network->outputLayer->nums;
    int idx = -1;
    double value = 0;

    for(int i = 0; i < network->layerSizes[network->numHiddenLayers + 1]; i++) {
        if (expected[i] > value) {
            idx = i;
            value = expected[i];

        }
    }
    
    result->expected = idx;

    value = outputs[0][0];
    idx = 0;
    for(int i = 0; i < network->layerSizes[network->numHiddenLayers + 1]; i++) {
        if (outputs[i][0] > value) {
            idx = i;
            value = outputs[i][0];
        }
    }
    result->predicted = idx;

    return result;
}

void saveWeights(Network* network, char* filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;


    for(int i = 0; i < network->numHiddenLayers + 1; i++) {
        int rows = network->weights[i]->rows;
        int cols = network->weights[i]->cols;

        fwrite(&rows, sizeof(int), 1, f);
        fwrite(&cols, sizeof(int), 1, f);

        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                fwrite(&network->weights[i]->nums[row][col], sizeof(double), 1, f);
            }
        }
    }



    fclose(f);
}

void saveBiases(Network* network, char* filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;


    for(int i = 0; i < network->numHiddenLayers + 1; i++) {
        int rows = network->biases[i]->rows;
        int cols = network->biases[i]->cols;

        fwrite(&rows, sizeof(int), 1, f);
        fwrite(&cols, sizeof(int), 1, f);

        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                fwrite(&network->biases[i]->nums[row][col], sizeof(double), 1, f);
            }
        }
    }



    fclose(f);
}

void loadWeights(Network* network, char* filename) {
    UNUSED(network);
    FILE *f = fopen(filename, "rb");
    if (!f) return;

    double num;
    int rows;
    int cols;

    // Matrix** ret = malloc(sizeof(Matrix*) * (network->numHiddenLayers + 1));


    for(int i = 0; i < network->numHiddenLayers + 1; i++) {
        fread(&rows, sizeof(int), 1, f);
        fread(&cols, sizeof(int), 1, f);

        double *nums = malloc(sizeof(double) * rows * cols);
        
        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                fread(&num, sizeof(double), 1, f);
                nums[row * cols + col] = num;
            }
        }
        destroyMatrix(network->weights[i]);
        network->weights[i] = createMatrix(rows, cols, nums);
        free(nums);

    }

    fclose(f);
    // return ret;
    
}

void loadBiases(Network* network, char* filename) {
    FILE *f = fopen(filename, "rb");

    double num;
    int rows;
    int cols;

    for(int i = 0; i < network->numHiddenLayers + 1; i++) {
        fread(&rows, sizeof(int), 1, f);
        fread(&cols, sizeof(int), 1, f);

        double *nums = malloc(sizeof(double) * rows * cols);
        
        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                fread(&num, sizeof(double), 1, f);
                nums[row * cols + col] = num;
            }
        }
        destroyMatrix(network->biases[i]);
        network->biases[i] = createMatrix(rows, cols, nums);
        free(nums);

    }

    fclose(f);
    
}

double evaluate(Example** examples, Network* network, int numData) {
    int correct = 0; 
    int total = 0;
    for (int i = 0; i < numData; i++) {
        destroyMatrix(network->input);
        network->input = unroll(examples[i]->input);
        network->expected = examples[i]->expected;
        forward(network);
        Result* result = argmax(network);
        if (result->expected == result->predicted) correct++; //
        printf("predicted: %d, expected: %d\n", result->predicted, result->expected);
        printMatrix2(examples[i]->input);
        free(result);
        total++; 

    }
    return (((double)correct)/total) * 100;


}

