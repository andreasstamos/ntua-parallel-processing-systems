.KEEP_STATE:

#MACROS = -DTEST_CONV
#MACROS = -DPRINT_RESULTS

MPICC = mpicc

CFLAGS = -Wall -Wextra -Wno-unused -O3 -std=gnu99 ${MACROS} 

LDFLAGS = -lm

all: jacobi_mpi gauss_mpi redblack_mpi

jacobi_mpi: jacobi_mpi.o utils.o
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

gauss_mpi: gauss_mpi.o utils.o
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

redblack_mpi: redblack_mpi.o utils.o
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(MPICC) $(CFLAGS) -c $< -o $@
clean:
	rm -rf *.o jacobi_mpi gauss_mpi redblack_mpi

