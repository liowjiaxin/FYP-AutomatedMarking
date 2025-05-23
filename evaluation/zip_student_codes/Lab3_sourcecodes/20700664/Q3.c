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
    Student* newStudent = (Student*)malloc(sizeof(Student)); // dynamic allocate memory for newStudent
    strcpy(newStudent->name, name);
    newStudent->id = id;
    newStudent->grade = grade;
    newStudent->next = NULL; // initialize "next" to NULL
    return newStudent;
}

// Function to insert a new student record
void insertStudent(const char* name, int id, float grade) {
    Student* newStudent = createStudent(name, id, grade);
    newStudent->next = head; // link "next" to the start of the linked list
    head = newStudent; // update head pointer
}

// Function to display all student records
void displayStudents() {
    Student* current = head; // start at the head of the linked list
    if(current != NULL){ 
        while (current != NULL) {
            printf("Name: %s, ID: %d, Grade: %.2f\n", current->name, current->id, current->grade);
            current = current->next;
        }
    } else {
        printf("Student records not found.\n");
    }
}

// Function to search for a student by ID
Student* searchStudentByID(int id) {
    Student* current = head;
    while (current != NULL) {
        if (current->id == id) { // when current ID matches, display the student record
            return current;
        }
        current = current->next;
    }
    return NULL;
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
}

void freeMemory() {
    deleteList();
}

// Sort function to sort the student records by grade based on an algorithm of your choice
void sortStudents() {
    if (head == NULL) { // if head == NULL exit
    return;
    }

    Student* i, *j;
    char tempName[100];
    int tempID;
    float tempGrade;

    for (i = head; i != NULL; i = i->next) { // outer loop
        for (j = i->next; j != NULL; j = j->next) { // inner loop
            if (i->grade > j->grade) { // if grades are not in order, then swap student records
                strcpy(tempName, i->name);// swap name
                strcpy(i->name, j->name);
                strcpy(j->name, tempName);

                tempID = i->id;// swap ID
                i->id = j->id;
                j->id = tempID;

                tempGrade = i->grade;// swap grade
                i->grade = j->grade;
                j->grade = tempGrade;
            }
        }
    }
    printf("Records sorted by grade using Bubble Sort.\n");

} 

// Main function to drive the program
int main() {
    int choice, id;
    char name[100];
    float grade;
    do {
        printf("\n1. Insert Student Record\n");
        printf("2. Display Student Records\n");
        printf("3. Sort Records\n");
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
            strtok(name, "\n");
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
                printf("Found: Name: %s, ID: %d, Grade: %.2f\n",
                found->name, found->id, found->grade);
            } else {
                printf("Student not found.\n");
            }
            break;
        case 5:
            deleteList();
            printf("List deleted.\n");
            break;
        case 6:
            freeMemory();
            printf("Exiting...\n");
            break;
        default:
            printf("Invalid choice! Please try again.\n");
        }
    } while (choice != 6);
 return 0;
}