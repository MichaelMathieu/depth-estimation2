CXX=g++
LIBS=`pkg-config --libs opencv`
FLAGS=-W -fpic -O3 -funroll-loops -fopenmp
ARCH=$(shell bash -c 'if [[ `uname -a` =~ "x86" ]]; then echo "__x86__"; else echo "__ARM__"; fi')

all: oextract

.cpp.o:
	${CXX} -D${ARCH} ${FLAGS} -c $< -o $@

oextract: matching.cpp matching.o
	${CXX} ${FLAGS} matching.o ${LIBS} -shared -o libmatching.so

clean:
	rm -rf libmatching.so matching.o