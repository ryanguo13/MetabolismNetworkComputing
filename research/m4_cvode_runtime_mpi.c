#include <mpi.h>
#include <sundials/sundials_context.h>

int m4_cvode_mpi_initialize(void)
{
    int initialized = 0;
    int status = MPI_Initialized(&initialized);
    if (status != MPI_SUCCESS || initialized) return status;
    return MPI_Init(NULL, NULL);
}

int m4_cvode_mpi_context_create(SUNContext *context)
{
    return SUNContext_Create(MPI_COMM_WORLD, context);
}

int m4_cvode_mpi_finalize(void)
{
    int finalized = 0;
    int status = MPI_Finalized(&finalized);
    if (status != MPI_SUCCESS || finalized) return status;
    return MPI_Finalize();
}
