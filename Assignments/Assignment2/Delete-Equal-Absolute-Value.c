#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct Node* Link;
struct Node {
    int data;
    Link next;
};

int main()
{
    int m, n;
    scanf("%d%d", &m, &n);
    int tag[n];
    memset(tag, 0, sizeof(int)*n);

    Link Head = (Link)malloc(sizeof(struct Node));
    Head->next = NULL;

    Link ptr = Head;

    for (int i = 0; i < m; i++)
    {
        int t;
        scanf("%d", &t);
        Link temp = (Link)malloc(sizeof(struct Node));
        temp->data = t;
        temp->next = NULL;
        ptr->next = temp;
        ptr = ptr->next;
    }

    printf("Original link:");
    for (ptr = Head->next; ptr != NULL; ptr = ptr->next)
    {
        printf(" %d", ptr->data);
    }
    printf("\n");

    for (ptr = Head; ptr->next != NULL; ptr = ptr->next)
    {
        Link t;
        for (t = ptr->next; t != NULL && tag[abs(t->data)] == 1; t = t->next)
        {
                
        } 
        ptr->next = t;
        if (t != NULL)
        {
            tag[abs(t->data)] = 1;
        }
        else break;
    }

    printf("Deleted link:");
    for (ptr = Head->next; ptr != NULL; ptr = ptr->next)
    {
        printf(" %d", ptr->data);
    }
    printf("\n");

    return 0;
}