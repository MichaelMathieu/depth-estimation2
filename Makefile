CXX=g++
LIBS=`pkg-config --libs opencv`
FLAGS=-W -fpic -O3 -funroll-loops -fopenmp

all: oextract

oextract:
	${CXX} ${FLAGS} matching.cpp ${LIBS} -shared -o libmatching.so

clean:
	rm -rf libmatching.so