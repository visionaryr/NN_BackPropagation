# build an executable named myprog from myprog.c
# the compiler: gcc for C program, define as g++ for C++
CC = g++

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS  = -g -Wall

# the build target executable:
TARGET = matrixprog

$(TARGET): main.cpp
	$(CC) $(CFLAGS) -o $(TARGET) -std=c++11 main.cpp matrix.cpp png.cpp matrix_calculate.cpp network.cpp bp_func.cpp dataset.cpp -lpng
	./$(TARGET)

clean:
	$(RM) $(TARGET)
