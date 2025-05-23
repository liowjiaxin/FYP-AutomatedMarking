#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Structure for Student
typedef struct Student {
    char name[100];
    int id;
    float grade;
    struct Student* next;
} Student;

Student* head = NULL;

// Function to create a new student node
Student* createStudent(const char* name, int id, float grade) {
    Student* newStudent = (Student*)malloc(sizeof(Student));
    if (!newStudent) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    strcpy(newStudent->name, name);
    newStudent->id = id;
    newStudent->grade = grade;
    newStudent->next = NULL;
    return newStudent;
}

// Function to insert a new student record
void insertStudent(const char* name, int id, float grade) {
    Student* newStudent = createStudent(name, id, grade);
    newStudent->next = head;
    head = newStudent;
    printf("Student record added successfully!\n");
}

// Function to display all student records
void displayStudents() {
    if (head == NULL) {
        printf("No student records found.\n");
        return;
    }

    Student* current = head;
    printf("Student Records:\n");
    while (current != NULL) {
        printf("Name: %s, ID: %d, Grade: %.2f\n", current->name, current->id, current->grade);
        current = current->next;
    }
}

// Function to delete the entire list
void deleteList() {
    Student* current = head;
    Student* next;

    while (current != NULL) {
        next = current->next;
        free(current);
        current = next;
    }
    head = NULL;
    printf("All records deleted successfully!\n");
}

void freeMemory() {
    deleteList();
}

// Sort function to sort the student records by grade using Bubble Sort
void sortStudents() {
    if (head == NULL || head->next == NULL) {
        printf("Not enough records to sort.\n");
        return;
    }

    Student* current;
    Student* next;
    int swapped;

    do {
        swapped = 0;
        current = head;

        while (current->next != NULL) {
            next = current->next;
            if (current->grade > next->grade) {
                // Swap data
                char tempName[100];
                int tempID;
                float tempGrade;

                strcpy(tempName, current->name);
                tempID = current->id;
                tempGrade = current->grade;

                strcpy(current->name, next->name);
                current->id = next->id;
                current->grade = next->grade;

                strcpy(next->name, tempName);
                next->id = tempID;
                next->grade = tempGrade;

                swapped = 1;
            }
            current = current->next;
        }
    } while (swapped);

    printf("Records sorted by grade using Bubble Sort.\n");
}

// Function to search for a student record by ID
Student* searchStudentByID(int id) {
    Student* current = head;
    while (current != NULL) {
        if (current->id == id) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

// Main function to drive the program
int main() {
    int choice, id;
    char name[100];
    float grade;

    do {
        printf("\n1. Insert Student Record\n");
        printf("2. Display Student Record\n");
        printf("3. Sort Record\n");
        printf("4. Search Record by ID\n");
        printf("5. Delete List\n");
        printf("6. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        getchar(); // To consume the newline character

        switch (choice) {
            case 1:
                printf("Enter Name: ");
                fgets(name, sizeof(name), stdin);
                strtok(name, "\n"); // Remove newline character
                printf("Enter ID: ");
                scanf("%d", &id);
                printf("Enter Grade: ");
                scanf("%f", &grade);
                insertStudent(name, id, grade);
                break;
            case 2:
                displayStudents();
                break;
            case 3:
                sortStudents();
                break;
            case 4:
                printf("Enter ID to search: ");
                scanf("%d", &id);
                Student* found = searchStudentByID(id);
                if (found) {
                    printf("Found: Name: %s, ID: %d, Grade: %.2f\n", found->name, found->id, found->grade);
                } else {
                    printf("Student not found.\n");
                }
                break;
            case 5:
                deleteList();
                break;
            case 6:
                freeMemory();
                printf("Exiting.\n");
                break;
            default:
                printf("Invalid choice! Please try again.\n");
        }
    } while (choice != 6);

    return 0;
}
