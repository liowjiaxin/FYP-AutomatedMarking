#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

#define MIN_BID 100000
#define MIN_BIDS 2

int main3()//When using, remove the redundant characters in the file so as not to affect the program 
{
    int N;

    // ���뾺����������֤
    printf("Enter the number of bids: ");
    scanf("%d", &N);

    while (N < MIN_BIDS)
    {
        printf("Error: Number of bids must be at least %d. Please try again: ", MIN_BIDS);
        scanf("%d", &N);
    }

    int* bids = (int*)malloc(N * sizeof(int));
    if (bids == NULL)
    {
        printf("Memory allocation failed!\n");
        return 1; // �����˳�
    }

    int highestBid = 0; // ��ʼ����߾���ֵ

    printf("Enter the bids (must be at least $%d):\n", MIN_BID);
    for (int i = 0; i < N; i++)
    {
        printf("Bid %d: ", i + 1);
        scanf("%d", &bids[i]);

        while (bids[i] < MIN_BID)
        {
            printf("Error: Bid must be at least $%d. Please try again: ", MIN_BID);
            scanf("%d", &bids[i]);
        }

        // ������߾���ֵ
        if (bids[i] > highestBid)
        {
            highestBid = bids[i];
        }
    }

    // �����߾���ֵ
    printf("The highest bid is: $%d\n", highestBid);


    return 0;
}