.KEEP_STATE:

CC = gcc

CFLAGS = -Wall -Wextra -Wno-unused -O3

LDFLAGS = 

H_FILES = kmeans.h

COMM_SRC = file_io.c util.c


all: kmeans_seq

kmeans_seq: main.o file_io.o util.o seq_kmeans.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.c $(H_FILES)
	$(CC) $(CFLAGS) -c $< -o $@

seq_kmeans.o: seq_kmeans.c $(COMM_SRC) $(H_FILES)
	$(CC) $(CFLAGS) -c $< -o $@

file_io.o: file_io.c
	$(CC) $(CFLAGS) -c $< -o $@

util.o: util.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o kmeans_seq

