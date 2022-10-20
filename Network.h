#define UNUSED(a) ((void) a)
typedef struct {
    Matrix *input;
    int *expected;

} Example;


typedef struct {
    Matrix *input;          // User specified matrix. we will unroll.
    int numHiddenLayers;    // User specified value of how many hidden layers there are in the network
    int *layerSizes;        // Array to hold the size of each layer, including the output layer
    Matrix *outputLayer;    // Array of size layersize[numHiddenLayers] that holds the predictions
    Matrix **hiddenLayers;  // Array size numHiddenLayers of Matrix size layerSizes[i] by 1 for i in layerSizes
    Matrix **weights;       // Array size numHiddenLayers + 1 of Matrix size layerSizes[i] by layerSizes[i + 1] for i in layerSizes
    Matrix **biases;        // Array size numHiddenLayers of Matrix size layerSizes[i] by 1 for i in layerSizes
    int *expected;          // Array size layerSize[numHiddenLayers] that holds the actual expected output
    double learningRate;    // Hyperparameter: Determines how big the step size will be 
    Example** exampleList;  // Training Data in the form of a list of pointers to struct Example
    int epochs;             // How many rounds of training
    int miniBatchSize;      // How big each mini batch will be in stochastic gradient descent

} Network;

typedef struct {
    Matrix **delWeights;     // Gradient for the cost function for the weights
    Matrix **delBiases;      // Gradient for the cost function for the biases
} Gradient;

typedef struct {
    int predicted;
    int expected;
} Result;

Network* createNetwork(Example **exampleList, int numHiddenLayers, int *layerSizes, double learningRate, int epochs, int miniBatchSize);

void destroyNetwork(Network* network);

void forward(Network *network);

Matrix** backpropForward(Network *network);

double loss(Network *network);

Matrix* lossDerivative(Network *network);

void shuffle(Example** exampleList, int numData);

void SGD(Network *network, int numData);

void updateMiniBatch(Network *network, Example** miniBatch);

Gradient *backprop(Network* network, Example* example);

Example** createExamples(int numData, Matrix** inputList, int *labelList, int numClasses);

Example** createExamplesFromFile(int numData, int numClasses, char* filename);

Result* argmax(Network* network);  

void saveWeights(Network* network, char* filename);

void loadWeights(Network* network, char* filename);

void saveBiases(Network* network, char* filename);

void loadBiases(Network* network, char* filename);

void destroyExamples(Example** examples, int numData);

double evaluate(Example** examples, Network* network, int numData);
