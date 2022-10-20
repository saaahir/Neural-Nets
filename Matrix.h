typedef struct {
  int rows;
  int cols;
  double** nums;
} Matrix;

// Initializations 
Matrix* createMatrix(int numRows, int numCols, double *nums);

Matrix* copyMatrix(Matrix *matrix);

void destroyMatrix(Matrix *matrix);

void printMatrix(Matrix *matrix);

void printMatrix2(Matrix *matrix);

void randomizeMatrix(Matrix *matrix, int lower, int upper);

void setMatrix(Matrix *matrix, double num); 

void writeMatrix(char *filename);

Matrix *readMatrix(char *filename);

// Operations 
Matrix* transpose(Matrix *matrix);

Matrix* dot(Matrix *leftMatrix, Matrix *rightMatrix);

double totalSum(Matrix *matrix);

double *rowSum(Matrix *matrix);

double *colSum(Matrix *matrix);

void applyFunction(Matrix *matrix, double (*func)(double, double), double arg);

void applyRowFunction(Matrix *matrix, double (*func)(double, double), double *arg);

Matrix* unroll(Matrix *matrix);

void add(Matrix *first, Matrix *second);

void subtract(Matrix *first, Matrix *second);

void multiply(Matrix *first, Matrix *second);

void assertMatrixEquals(Matrix *first, Matrix *second);

void scale(Matrix *first, double scaler);
