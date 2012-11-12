CXX=g++
LIBS=`pkg-config --libs opencv`
FLAGS=-W -fpic -O3 -funroll-loops -fopenmp
#FLAGS=-W -fpic -O0
ARCH=$(shell bash -c 'if [[ `uname -a` =~ "x86" ]]; then echo "__x86__"; else echo "__ARM__"; fi')
ifeq (${ARCH}, __ARM__)
FLAGS+= -mfpu=neon
endif

all: match filter

.cpp.o:
	${CXX} -D${ARCH} ${FLAGS} -c $< -o $@

match: matching.cpp matching.o
	${CXX} ${FLAGS} matching.o ${LIBS} -shared -o libmatching.so

filter: filtering.cpp filtering.o
	${CXX} ${FLAGS} filtering.o ${LIBS} -shared -o libfiltering.so

clean:
	rm -rf libmatching.so matching.o filtering.o libfiltering.so