#include <stdio.h>
#include <stdlib.h>

//#define DEBUG
#define NON_DEBUG

extern int compute_gold(float *, int);

/* Perform Gaussian elimination in place on the U matrix */
int compute_gold(float *U, int num_elements)
{
#ifdef NON_DEBUG
    int i, j, k;
    
    for (k = 0; k < num_elements; k++) {
        for (j = (k + 1); j < num_elements; j++) {   /* Reduce the current row. */
            if (U[num_elements * k + k] == 0) {
                fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                return -1;
            }            
	    /* Division step */
            U[num_elements * k + j] = (float)(U[num_elements * k + j] / U[num_elements * k + k]);	
        }

        U[num_elements * k + k] = 1;	/* Set the principal diagonal entry in U to 1 */ 
        
        for (i = (k + 1); i < num_elements; i++) {
            for (j = (k + 1); j < num_elements; j++)
                U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]);	/* Elimination step */
            
            U[num_elements * i + k] = 0;
        }
    }
    
    return 0;
#endif

#ifdef DEBUG
    int i, j, k, v;
    int c = 1;
    double g, h; 
    for (k = 0; k < num_elements; k++) {
	printf("\n\n|%.2f -> 1.00| ", U[num_elements * k + k]);
        for (j = (k + 1); j < num_elements; j++) {   /* Reduce the current row. */
            if (U[num_elements * k + k] == 0) {
                fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                return -1;
            }            
	    /* Division step */
	    h = U[num_elements * k + j];
            U[num_elements * k + j] = (float)(U[num_elements * k + j] / U[num_elements * k + k]);	
	    printf("|%.2f -> %.2f| ", h, U[num_elements * k + j]);
	    
        }
	
        U[num_elements * k + k] = 1;	/* Set the principal diagonal entry in U to 1 */ 
        printf("\nElimination step\n"); 
        for (i = (k + 1); i < num_elements; i++) {
	    for (v = 0; v < c; v++){ 
		    printf("|%.2f -> 0.00| ", U[num_elements * i + k]);
	    }
            for (j = (k + 1); j < num_elements; j++) {
		g = U[num_elements * i + j];	
                U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]);	/* Elimination step */
		printf("|%.2f -> %.2f| ", g, U[num_elements * i + j]);
	    }
	    printf("\n");
            
            U[num_elements * i + k] = 0;
        }
	c++;
    }
    
    return 0;
#endif
}
