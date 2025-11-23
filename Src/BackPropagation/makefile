# ==============================================================================
# 1. Configuration Variables
# ==============================================================================

# Executable name
TARGET = BpProgram

# Directories
SRC_DIR = .
BIN_DIR = bin
OBJ_DIR = obj

# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -g # -g for debugging info
LDFLAGS = -lpng

# ==============================================================================
# 2. File Lists and Derived Paths
# ==============================================================================

# Exclude files list
EXCLUDE_FILES = $(SRC_DIR)/bp_func.cpp

# Find all source files in the current directory (or $(SRC_DIR))
SRCS = $(filter-out $(EXCLUDE_FILES), $(wildcard $(SRC_DIR)/*.cpp))

# Map source files to their corresponding object file paths in the OBJ_DIR
# Example: main.cpp -> obj/main.o
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Full path to the final executable
EXEC = $(BIN_DIR)/$(TARGET)

# ==============================================================================
# 3. Core Build Targets
# ==============================================================================

# Default target: builds the executable
.PHONY: all
all: $(EXEC)

# Debug target: Add -DDEBUG_ENABLED to CXXFLAGS to enable debug messages.
.PHONY: debug
debug: CXXFLAGS += -DDEBUG_ENABLED
debug: $(EXEC)

# Rule to Link the final executable (The link step)
# Dependencies: All object files (OBJS)
# Order-Only Prerequisite: The bin directory must exist (| $(BIN_DIR))
$(EXEC): $(OBJS) | $(BIN_DIR)
	@echo "--- Linking executable: $(EXEC) ---"
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

# Rule to Compile C++ source files into object files
# This is a pattern rule: %.o is the target, %.cpp is the prerequisite
# Dependencies: Source file (main.cpp)
# Order-Only Prerequisite: The obj directory must exist (| $(OBJ_DIR))
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "   Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ==============================================================================
# 4. Directory Management Rules (Using the Order-Only Prerequisite Pattern)
# ==============================================================================

# Rule to create the bin directory
# The `mkdir -p` command ensures it only runs if the directory doesn't exist.
$(BIN_DIR):
	@echo "Creating output directory: $(BIN_DIR)"
	mkdir -p $(BIN_DIR)

# Rule to create the obj directory
$(OBJ_DIR):
	@echo "Creating intermediate directory: $(OBJ_DIR)"
	mkdir -p $(OBJ_DIR)

# ==============================================================================
# 5. Utility Targets
# ==============================================================================

# Cleans all generated files and directories
.PHONY: clean
clean:
	@echo "--- Cleaning up build directories ---"
	$(RM) -r $(BIN_DIR) $(OBJ_DIR)

# Runs the compiled program
.PHONY: run
run: $(EXEC)
	@echo "--- Running $(TARGET) ---"
	./$(EXEC)