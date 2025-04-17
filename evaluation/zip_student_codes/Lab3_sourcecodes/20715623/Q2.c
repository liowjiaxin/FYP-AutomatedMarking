#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#define MAX_ITEMS 100

// �����ӻ���Ŀ�Ľṹ��
typedef struct 
{
    char name[100]; // ����
    char category[50]; // ���
    int quantity; // ����
    float price; // �۸�
    bool inStock; // �Ƿ��п��
} GroceryItem;

// �ӻ����
GroceryItem inventory[MAX_ITEMS];
int itemCount = 0; // ��ǰ����е���Ʒ����

// ��������
void addGroceryItem(); // ����ӻ���Ŀ
void listGroceryItems(); // �г������ӻ���Ŀ
void updateQuantity(); // �����ӻ���Ŀ����
void removeGroceryItem(); // �Ƴ��ӻ���Ŀ

int main2() //ʹ��ʱɾ��main����ַ��������޷�����
{
    int choice;
    do
    {
        // ��ӡ�˵�
        printf("\nGrocery Inventory Menu:\n");
        printf("1. Add Grocery Item\n");
        printf("2. List All Grocery Items\n");
        printf("3. Update Quantity\n");
        printf("4. Remove Grocery Item\n");
        printf("5. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) 
        {
        case 1:
            addGroceryItem();
            break;
        case 2:
            listGroceryItems();
            break;
        case 3:
            updateQuantity();
            break;
        case 4:
            removeGroceryItem();
            break;
        case 5:
            printf("Exiting...\n");
            break;
        default:
            printf("Invalid choice. Please try again.\n");
        }
    } while (choice != 5);

    return 0;
}

// ����ӻ���Ŀ
void addGroceryItem()
{
    if (itemCount >= MAX_ITEMS) 
    {
        printf("Inventory is full. Cannot add more items.\n");
        return;
    }

    GroceryItem newItem;

    printf("Enter item name: ");
    getchar(); // ���������
    fgets(newItem.name, sizeof(newItem.name), stdin);
    strtok(newItem.name, "\n"); // �Ƴ����з�

    printf("Enter category: ");
    fgets(newItem.category, sizeof(newItem.category), stdin);
    strtok(newItem.category, "\n"); // �Ƴ����з�

    printf("Enter quantity: ");
    scanf("%d", &newItem.quantity);

    printf("Enter price: ");
    scanf("%f", &newItem.price);

    newItem.inStock = newItem.quantity > 0; // �����������ÿ��״̬

    inventory[itemCount++] = newItem; // ��ӵ����
    printf("Item added successfully!\n");
}

// �г������ӻ���Ŀ
void listGroceryItems() 
{
    if (itemCount == 0) 
    {
        printf("Inventory is empty.\n");
        return;
    }

    printf("\n%-20s %-15s %-10s %-10s %-10s\n", "Name", "Category", "Quantity", "Price", "In Stock");
    printf("-------------------------------------------------------------\n");
    for (int i = 0; i < itemCount; i++) 
    {
        printf("%-20s %-15s %-10d $%-9.2f %-10s\n",
            inventory[i].name,
            inventory[i].category,
            inventory[i].quantity,
            inventory[i].price,
            inventory[i].inStock ? "Yes" : "No");
    }
}

// �����ӻ���Ŀ����
void updateQuantity() 
{
    char name[100];
    printf("Enter the name of the item to update: ");
    getchar(); // ���������
    fgets(name, sizeof(name), stdin);
    strtok(name, "\n"); // �Ƴ����з�

    for (int i = 0; i < itemCount; i++) 
    {
        if (strcmp(inventory[i].name, name) == 0)
        {
            printf("Enter new quantity: ");
            scanf("%d", &inventory[i].quantity);
            inventory[i].inStock = inventory[i].quantity > 0; // ���¿��״̬
            printf("Quantity updated successfully!\n");
            return;
        }
    }

    printf("Item not found.\n");
}

// �Ƴ��ӻ���Ŀ
void removeGroceryItem() 
{
    char name[100];
    printf("Enter the name of the item to remove: ");
    getchar(); // ���������
    fgets(name, sizeof(name), stdin);
    strtok(name, "\n"); // �Ƴ����з�

    for (int i = 0; i < itemCount; i++) 
    {
        if (strcmp(inventory[i].name, name) == 0)
        {
            // ������Ԫ��ǰ�Ƹ���Ҫ�Ƴ���Ԫ��
            for (int j = i; j < itemCount - 1; j++)
            {
                inventory[j] = inventory[j + 1];
            }
            itemCount--;
            printf("Item removed successfully!\n");
            return;
        }
    }

    printf("Item not found.\n");
}
