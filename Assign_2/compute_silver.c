#include <stdio.h>
#include <stdlib.h>

extern void *compute_silver(void *);

/* Perform Gaussian elimination in place on the U matrix */
void *compute_silver(void *args)
{
    int i, j, k;

    ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *) args;

    for (k = 0; k < args_for_me->num_elements; k++) {
        for (j = (k + 1); j < args_for_me->num_elements; j++) {   /* Reduce the current row. */
            if (args_for_me->U[args_for_me->num_elements * k + k] == 0) {
                fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                return -1;
            }            
	    /* Division step */
            args_for_me->U[args_for_me->num_elements * k + j] = (float)(args_for_me->U[args_for_me->num_elements * k + j] / args_for_me->U[args_for_me->num_elements * k + k]);	
        }

        args_for_me->U[args_for_me->num_elements * k + k] = 1;	/* Set the principal diagonal entry in U to 1 */ 
        
        for (i = (k + 1); i < args_for_me->num_elements; i++) {
            for (j = (k + 1); j < args_for_me->num_elements; j++)
                args_for_me->U[args_for_me->num_elements * i + j] = args_for_me->U[args_for_me->num_elements * i + j] - (args_for_me->U[args_for_me->num_elements * i + k] * args_for_me->U[args_for_me->num_elements * k + j]);	/* Elimination step */
            
            args_for_me->U[args_for_me->num_elements * i + k] = 0;
        }
    }
    
    return 0;
}
