CC = gcc

FLAGS = -fPIC -O3 -shared -Wl,-z,defs
LIBS = -llapack -lm

all: bdgeq.so

bdgeq.so:
	$(CC) ${FLAGS} -o bdgeq.so bdgeq.c $(LIBS)

clean:
	rm bdgeq.so
