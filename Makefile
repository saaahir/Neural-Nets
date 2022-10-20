CC = gcc

CFLAGS = -std=c99					# Using the C99 standard
CFLAGS += -Wall						# This enables all the warnings about constructions that some users consider questionable, and that are easy to avoid (or modify to prevent the warning), even in conjunction with macros
CFLAGS += -pedantic					# Issue all the warnings demanded by strict ISO C and ISO C++; reject all programs that use forbidden extensions, and some other programs that do not follow ISO C and ISO C++
CFLAGS += -Wextra					# This enables some extra warning flags that are not enabled by -Wall
CFLAGS += -Werror					# Make all warnings into errors
# CFLAGS += -O0						# Optimize even more. GCC performs nearly all supported optimizations that do not involve a space-speed tradeoff.
CFLAGS += -Wstrict-prototypes		# Warn if a function is declared or defined without specifying the argument types
CFLAGS += -Wold-style-definition	# Warn if an old-style function definition is used. A warning is given even if there is a previous prototype
# CFLAGS += -g						# Generate debugging information
# CFLAGS += -Werror=vla			    # Generate an error if variable-length arrays (bad practice in C!) are used

# Source files to be compiled together
# CFILES = Matrix.c Network.c Driver.c Test2.c
CFILES = Matrix.c Network.c Driver.c 
HFILES = Matrix.h Network.h

# OBJNAME = Test2 
OBJNAME = Driver

all: $(CFILES)
	@$(CC) $(CFLAGS) $(CFILES) -o $(OBJNAME)

Network: $(CFILES) $(HFILES)
	@$(CC) $(CFLAGS) $(CFILES) -o $@

Driver: $(CFILES) $(HFILES)
	@$(CC) $(CFLAGS) $(CFILES) -o $@

Test2: $(CFILES) $(HFILES)
	@$(CC) $(CFLAGS) $(CFILES) -o $@

.PHONY: clean
clean:
	@ rm -f $(OBJNAME) *.o *.out
