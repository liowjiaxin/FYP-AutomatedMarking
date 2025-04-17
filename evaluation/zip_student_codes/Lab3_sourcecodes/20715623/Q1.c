#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#define MAX 10

// ��������
void inputMatrix(int matrix[MAX][MAX], int rows, int cols); // �������Ԫ�صĺ���
void printMatrix(int matrix[MAX][MAX], int rows, int cols); // ��ӡ����ĺ���
void addMatrices(int A[MAX][MAX], int B[MAX][MAX], int result[MAX][MAX], int rows, int cols); // ����ӷ��ĺ���
void subtractMatrices(int A[MAX][MAX], int B[MAX][MAX], int result[MAX][MAX], int rows, int cols); // ��������ĺ���
void transposeMatrix(int matrix[MAX][MAX], int result[MAX][MAX], int rows, int cols); // �����ת�õĺ���

int main1() //ʹ��ʱɾ��main����ַ��������޷�����
{
    int A[MAX][MAX], B[MAX][MAX]; // ����A��B
    int rowsA, colsA, rowsB, colsB; // ����A��B��ά��
    int result[MAX][MAX]; // �洢����ľ���
    int choice; // �û��˵�ѡ�����

    // �������A��ά��
    printf("Enter the number of rows and columns for Matrix A (max 10x10): ");
    scanf("%d %d", &rowsA, &colsA);

    // �������B��ά��
    printf("Enter the number of rows and columns for Matrix B (max 10x10): ");
    scanf("%d %d", &rowsB, &colsB);

    // ��֤����ά���Ƿ�Ϸ�
    if (rowsA > MAX || colsA > MAX || rowsB > MAX || colsB > MAX || rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0)
    {
        printf("Invalid matrix dimensions. Please use dimensions between 1 and 10.\n");
        return 1; // ���ά����Ч�����˳�����
    }

    // �������A��Ԫ��
    printf("Enter elements for Matrix A:\n");
    inputMatrix(A, rowsA, colsA);

    // �������B��Ԫ��
    printf("Enter elements for Matrix B:\n");
    inputMatrix(B, rowsB, colsB);

    // ��������Ĳ˵���������
    do 
    {
        printf("\nMatrix Operations Menu:\n");
        printf("1. Add Matrices\n");
        printf("2. Subtract Matrices\n");
        printf("3. Transpose Matrix A\n");
        printf("4. Transpose Matrix B\n");
        printf("5. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) 
        {
        case 1:
            // ����A��B���
            if (rowsA == rowsB && colsA == colsB) 
            {
                addMatrices(A, B, result, rowsA, colsA);
                printf("Result of Matrix A + Matrix B:\n");
                printMatrix(result, rowsA, colsA);
            }
            else 
            {
                printf("Addition not possible. Matrices must have the same dimensions.\n");
            }
            break;

        case 2:
            // ����A��B���
            if (rowsA == rowsB && colsA == colsB) 
            {
                subtractMatrices(A, B, result, rowsA, colsA);
                printf("Result of Matrix A - Matrix B:\n");
                printMatrix(result, rowsA, colsA);
            }
            else 
            {
                printf("Subtraction not possible. Matrices must have the same dimensions.\n");
            }
            break;

        case 3:
            // ����A��ת��
            transposeMatrix(A, result, rowsA, colsA);
            printf("Transpose of Matrix A:\n");
            printMatrix(result, colsA, rowsA);
            break;

        case 4:
            // ����B��ת��
            transposeMatrix(B, result, rowsB, colsB);
            printf("Transpose of Matrix B:\n");
            printMatrix(result, colsB, rowsB);
            break;

        case 5:
            // �˳�����
            printf("Exiting...\n");
            break;

        default:
            printf("Invalid choice. Please try again.\n");
        }
    } while (choice != 5); // ѭ��ֱ���û�ѡ���˳�

    return 0;
}

// �������Ԫ�صĺ���
void inputMatrix(int matrix[MAX][MAX], int rows, int cols) 
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++) 
        {
            printf("Enter element [%d][%d]: ", i + 1, j + 1);
            scanf("%d", &matrix[i][j]);
        }
    }
}

// ��ӡ����ĺ���
void printMatrix(int matrix[MAX][MAX], int rows, int cols) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", matrix[i][j]); // ��ӡÿ��Ԫ��
        }
        printf("\n"); // ����
    }
}

// ����ӷ��ĺ���
void addMatrices(int A[MAX][MAX], int B[MAX][MAX], int result[MAX][MAX], int rows, int cols) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            result[i][j] = A[i][j] + B[i][j]; // ��ӦԪ�����
        }
    }
}

// ��������ĺ���
void subtractMatrices(int A[MAX][MAX], int B[MAX][MAX], int result[MAX][MAX], int rows, int cols) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            result[i][j] = A[i][j] - B[i][j]; // ��ӦԪ�����
        }
    }
}

// �����ת�õĺ���
void transposeMatrix(int matrix[MAX][MAX], int result[MAX][MAX], int rows, int cols) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            result[j][i] = matrix[i][j]; // ���н���
        }
    }
}
