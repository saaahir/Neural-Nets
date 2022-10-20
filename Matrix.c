#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "Matrix.h"

#define UNUSED(a) ((void) a)
#define NUMROWS 4
#define NUMCOLS 3

Matrix* createMatrix(int numRows, int numCols, double* nums) {
    Matrix *matrix = malloc(sizeof(Matrix));
    if (!matrix) return NULL;
    matrix->rows = numRows;
    matrix->cols = numCols;
    matrix->nums = malloc(sizeof(double*) * numRows);
    for (int i = 0; i < numRows; i++) {
        matrix->nums[i] = malloc(sizeof(double) * numCols);
        for (int j = 0; j < numCols; j++) {
            if (nums == NULL) matrix->nums[i][j] = 0;
            else matrix->nums[i][j] = nums[i * numCols +j];
        }
    }
    return matrix;
}

Matrix* copyMatrix(Matrix *matrix) {
    Matrix *newMatrix = malloc(sizeof(Matrix));
    if (!newMatrix) return NULL;
    newMatrix->rows = matrix->rows;
    newMatrix->cols = matrix->cols;
    newMatrix->nums = malloc(sizeof(double*) * newMatrix->rows);
    for (int i = 0; i < newMatrix->rows; i++) {
        newMatrix->nums[i] = malloc(sizeof(double) * newMatrix->cols);
        for (int j = 0; j < newMatrix->cols; j++) {
            newMatrix->nums[i][j] = matrix->nums[i][j]; 
        }
    }
    return newMatrix;
}

void destroyMatrix(Matrix *matrix) {
    if(!matrix) return;
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->nums[i]);
    }
    free(matrix->nums);
    free(matrix);
}

void printMatrix(Matrix *matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double **nums = matrix->nums;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3.0f ", nums[i][j]);
        }
        printf("\n");  
    }
    printf("\n");  
}

void printMatrix2(Matrix *matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double **nums = matrix->nums;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3.0f ", nums[i][j] * 255);
        }
        printf("\n");  
    }
    printf("\n");  
}

void randomizeMatrix(Matrix *matrix, int lower, int upper) {
    UNUSED(lower);
    UNUSED(upper);
    int rows = matrix->rows;
    int cols = matrix->cols;
    double **nums = matrix->nums;

    srand(time(0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // nums[i][j] = rand() % (upper - lower + 1) + lower;  
            nums[i][j] = -1 + 2*(((float)rand())/RAND_MAX);
        }
    }
}

void setMatrix(Matrix *matrix, double num) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double **nums = matrix->nums;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nums[i][j] = num;
        }
    }
}

void writeMatrix(char *filename) {
    UNUSED(filename);
}

Matrix *readMatrix(char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) return NULL;
    char num;
    fseek(f, 28 * 28, SEEK_SET);
    for(int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            fscanf(f, "%c", &num);
            printf("%c ", num);
        }
        printf("\n");

    }

    fclose(f);

    return NULL;


}

Matrix* transpose(Matrix *matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    Matrix *newMatrix = createMatrix(cols, rows, NULL);
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            newMatrix->nums[i][j] = matrix->nums[j][i];
        }
    }
    return newMatrix;

}

Matrix* dot(Matrix *leftMatrix, Matrix *rightMatrix) {
    assert(leftMatrix->cols == rightMatrix->rows);

    Matrix *newMatrix = createMatrix(leftMatrix->rows, rightMatrix->cols, NULL);

    for (int i = 0; i < leftMatrix->rows; i++) {
        for (int j = 0; j < rightMatrix->cols; j++) {
            for (int k = 0; k < rightMatrix->rows; k++) {
                newMatrix->nums[i][j] += leftMatrix->nums[i][k] * rightMatrix->nums[k][j];
            }
        }
    }

    return newMatrix;

}

double totalSum(Matrix *matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double **nums = matrix->nums;

    double sum = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += nums[i][j];
        }
    }
    return sum;
}

double* rowSum(Matrix *matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double **nums = matrix->nums;

    double *rowSum = malloc(sizeof(double) * rows);

    double sum = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += nums[i][j];
        }
        rowSum[i] = sum;
        sum = 0;
    }
    return rowSum;
}

double* colSum(Matrix *matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double **nums = matrix->nums;

    double *rowSum = malloc(sizeof(double) * cols);

    double sum = 0;
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            sum += nums[j][i];
        }
        rowSum[i] = sum;
        sum = 0;
    }
    return rowSum;
}

void applyFunction(Matrix *matrix, double (*func)(double, double), double arg) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double **nums = matrix->nums;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nums[i][j] = func(nums[i][j], arg);
        }
    }
}

void applyRowFunction(Matrix *matrix, double (*func)(double, double), double *arg) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double **nums = matrix->nums;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nums[i][j] = func(nums[i][j], arg[i]);
        }
    }
}

void add(Matrix *first, Matrix *second) {
    assert(first->rows == second->rows);
    assert(first->cols == second->cols);

    for (int i = 0; i < first->rows; i++) {
        for (int j = 0; j < first->cols; j++) {
            first->nums[i][j] += second->nums[i][j];
        }
    }
}

void subtract(Matrix *first, Matrix *second) {
    assert(first->rows == second->rows);
    assert(first->cols == second->cols);

    for (int i = 0; i < first->rows; i++) {
        for (int j = 0; j < first->cols; j++) {
            first->nums[i][j] -= second->nums[i][j];
        }
    }
}

void multiply(Matrix *first, Matrix *second) {
    assert(first->rows == second->rows);
    assert(first->cols == second->cols);

    for (int i = 0; i < first->rows; i++) {
        for (int j = 0; j < first->cols; j++) {
            first->nums[i][j] *= second->nums[i][j];
        }
    }
}

void assertMatrixEquals(Matrix *first, Matrix *second) {
    assert(first->rows == second->rows);
    assert(first->cols == second->cols);

    for (int i = 0; i < first->rows; i++) {
        for (int j = 0; j < first->cols; j++) {
            assert(first->nums[i][j] == second->nums[i][j]);
        }
    }
}

void scale(Matrix *first, double scaler) {
    // printf("bruh %f\n",scaler);
    for (int i = 0; i < first->rows; i++) {
        for (int j = 0; j < first->cols; j++) {
            first->nums[i][j] *= scaler;
        }
    }
}



Matrix* unroll(Matrix *matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double **nums = matrix->nums;

    double newNums[rows * cols];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            newNums[i * cols + j] = nums[i][j];
        }
    }
    return createMatrix(rows * cols, 1, newNums);
}

