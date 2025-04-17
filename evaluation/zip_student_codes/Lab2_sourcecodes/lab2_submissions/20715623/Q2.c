#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

#define NAME_LENGTH 50
#define STUDENT_COUNT 3

// ����ԭ��
float calculateAverage(const float marks[], int count);
void displayStudentData(const char names[][NAME_LENGTH], const float marks[], int count, float average);

int main2()//When using, remove the redundant characters in the file so as not to affect the program 
{
    char names[STUDENT_COUNT][NAME_LENGTH]; // �洢ѧ������
    float marks[STUDENT_COUNT]; // �洢ѧ���ɼ�
    float average;

    // ����ѧ�������ͳɼ�
    for (int i = 0; i < STUDENT_COUNT; i++) {
        printf("Enter the name of student %d: ", i + 1);
        scanf(" %[^\n]", names[i]); 
        printf("Enter the mark of student %d: ", i + 1);
        scanf("%f", &marks[i]);
    }

    // ����ƽ���ɼ�
    average = calculateAverage(marks, STUDENT_COUNT);

    // ��ʾѧ�����ݺ�ƽ���ɼ�
    displayStudentData(names, marks, STUDENT_COUNT, average);

    return 0;
}

// ����ƽ���ֺ���
float calculateAverage(const float marks[], int count) {
    float sum = 0.0;

    for (int i = 0; i < count; i++) {
        sum += marks[i];
    }

    return sum / count;
}

// ��ʾѧ�����ݺ�ƽ���ֺ���
void displayStudentData(const char names[][NAME_LENGTH], const float marks[], int count, float average) {
    printf("\nStudent Names and Marks:\n");

    for (int i = 0; i < count; i++) {
        printf("%s: %.2f\n", names[i], marks[i]);
    }

    printf("\nAverage Mark: %.2f\n", average);
}