Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls   s/call   s/call  name    
 38.90      0.78     0.78        1     0.78     1.00  Matrix::sum_abs_cols()
 20.95      1.20     0.42                             main
 16.96      1.54     0.34 300010000     0.00     0.00  Matrix::operator[](int)
 10.47      1.75     0.21 200000000     0.00     0.00  std::abs(float)
  9.97      1.95     0.20        1     0.20     0.42  Matrix::sum_abs_rows()
  2.99      2.01     0.06        1     0.06     0.06  Matrix::print_matrix()
  0.00      2.01     0.00    10000     0.00     0.00  std::setw(int)
  0.00      2.01     0.00        3     0.00     0.00  Matrix::Matrix(int, int)
  0.00      2.01     0.00        3     0.00     0.00  Matrix::~Matrix()
  0.00      2.01     0.00        2     0.00     0.00  Matrix::sum_abs_matrix()
  0.00      2.01     0.00        1     0.00     0.00  _GLOBAL__sub_I_main
  0.00      2.01     0.00        1     0.00     0.00  parse_command_line(int, char**, unsigned int&, unsigned int&, unsigned long&, timeval&, int&)
  0.00      2.01     0.00        1     0.00     0.00  __static_initialization_and_destruction_0(int, int)
  0.00      2.01     0.00        1     0.00     0.00  std::setprecision(int)

 %         the percentage of the total running time of the
time       program used by this function.

cumulative a running sum of the number of seconds accounted
 seconds   for by this function and those listed above it.

 self      the number of seconds accounted for by this
seconds    function alone.  This is the major sort for this
           listing.

calls      the number of times this function was invoked, if
           this function is profiled, else blank.

 self      the average number of milliseconds spent in this
ms/call    function per call, if this function is profiled,
	   else blank.

 total     the average number of milliseconds spent in this
ms/call    function and its descendents per call, if this
	   function is profiled, else blank.

name       the name of the function.  This is the minor sort
           for this listing. The index shows the location of
	   the function in the gprof listing. If the index is
	   in parenthesis it shows where it would appear in
	   the gprof listing if it were to be printed.

Copyright (C) 2012-2021 Free Software Foundation, Inc.

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.

		     Call graph (explanation follows)


granularity: each sample hit covers 2 byte(s) for 0.50% of 2.01 seconds

index % time    self  children    called     name
                                                 <spontaneous>
[1]    100.0    0.42    1.59                 main [1]
                0.78    0.22       1/1           Matrix::sum_abs_cols() [2]
                0.20    0.22       1/1           Matrix::sum_abs_rows() [3]
                0.11    0.00 100000000/300010000     Matrix::operator[](int) [4]
                0.06    0.00       1/1           Matrix::print_matrix() [6]
                0.00    0.00       3/3           Matrix::~Matrix() [15]
                0.00    0.00       2/2           Matrix::sum_abs_matrix() [16]
                0.00    0.00       1/1           parse_command_line(int, char**, unsigned int&, unsigned int&, unsigned long&, timeval&, int&) [18]
                0.00    0.00       1/3           Matrix::Matrix(int, int) [14]
-----------------------------------------------
                0.78    0.22       1/1           main [1]
[2]     49.7    0.78    0.22       1         Matrix::sum_abs_cols() [2]
                0.11    0.00 100010000/300010000     Matrix::operator[](int) [4]
                0.11    0.00 100000000/200000000     std::abs(float) [5]
                0.00    0.00       1/3           Matrix::Matrix(int, int) [14]
-----------------------------------------------
                0.20    0.22       1/1           main [1]
[3]     20.8    0.20    0.22       1         Matrix::sum_abs_rows() [3]
                0.11    0.00 100000000/300010000     Matrix::operator[](int) [4]
                0.11    0.00 100000000/200000000     std::abs(float) [5]
                0.00    0.00       1/3           Matrix::Matrix(int, int) [14]
-----------------------------------------------
                0.11    0.00 100000000/300010000     main [1]
                0.11    0.00 100000000/300010000     Matrix::sum_abs_rows() [3]
                0.11    0.00 100010000/300010000     Matrix::sum_abs_cols() [2]
[4]     16.9    0.34    0.00 300010000         Matrix::operator[](int) [4]
-----------------------------------------------
                0.11    0.00 100000000/200000000     Matrix::sum_abs_rows() [3]
                0.11    0.00 100000000/200000000     Matrix::sum_abs_cols() [2]
[5]     10.4    0.21    0.00 200000000         std::abs(float) [5]
-----------------------------------------------
                0.06    0.00       1/1           main [1]
[6]      3.0    0.06    0.00       1         Matrix::print_matrix() [6]
                0.00    0.00   10000/10000       std::setw(int) [13]
                0.00    0.00       1/1           std::setprecision(int) [20]
-----------------------------------------------
                0.00    0.00   10000/10000       Matrix::print_matrix() [6]
[13]     0.0    0.00    0.00   10000         std::setw(int) [13]
-----------------------------------------------
                0.00    0.00       1/3           main [1]
                0.00    0.00       1/3           Matrix::sum_abs_rows() [3]
                0.00    0.00       1/3           Matrix::sum_abs_cols() [2]
[14]     0.0    0.00    0.00       3         Matrix::Matrix(int, int) [14]
-----------------------------------------------
                0.00    0.00       3/3           main [1]
[15]     0.0    0.00    0.00       3         Matrix::~Matrix() [15]
-----------------------------------------------
                0.00    0.00       2/2           main [1]
[16]     0.0    0.00    0.00       2         Matrix::sum_abs_matrix() [16]
-----------------------------------------------
                0.00    0.00       1/1           __libc_csu_init [42]
[17]     0.0    0.00    0.00       1         _GLOBAL__sub_I_main [17]
                0.00    0.00       1/1           __static_initialization_and_destruction_0(int, int) [19]
-----------------------------------------------
                0.00    0.00       1/1           main [1]
[18]     0.0    0.00    0.00       1         parse_command_line(int, char**, unsigned int&, unsigned int&, unsigned long&, timeval&, int&) [18]
-----------------------------------------------
                0.00    0.00       1/1           _GLOBAL__sub_I_main [17]
[19]     0.0    0.00    0.00       1         __static_initialization_and_destruction_0(int, int) [19]
-----------------------------------------------
                0.00    0.00       1/1           Matrix::print_matrix() [6]
[20]     0.0    0.00    0.00       1         std::setprecision(int) [20]
-----------------------------------------------

 This table describes the call tree of the program, and was sorted by
 the total amount of time spent in each function and its children.

 Each entry in this table consists of several lines.  The line with the
 index number at the left hand margin lists the current function.
 The lines above it list the functions that called this function,
 and the lines below it list the functions this one called.
 This line lists:
     index	A unique number given to each element of the table.
		Index numbers are sorted numerically.
		The index number is printed next to every function name so
		it is easier to look up where the function is in the table.

     % time	This is the percentage of the `total' time that was spent
		in this function and its children.  Note that due to
		different viewpoints, functions excluded by options, etc,
		these numbers will NOT add up to 100%.

     self	This is the total amount of time spent in this function.

     children	This is the total amount of time propagated into this
		function by its children.

     called	This is the number of times the function was called.
		If the function called itself recursively, the number
		only includes non-recursive calls, and is followed by
		a `+' and the number of recursive calls.

     name	The name of the current function.  The index number is
		printed after it.  If the function is a member of a
		cycle, the cycle number is printed between the
		function's name and the index number.


 For the function's parents, the fields have the following meanings:

     self	This is the amount of time that was propagated directly
		from the function into this parent.

     children	This is the amount of time that was propagated from
		the function's children into this parent.

     called	This is the number of times this parent called the
		function `/' the total number of times the function
		was called.  Recursive calls to the function are not
		included in the number after the `/'.

     name	This is the name of the parent.  The parent's index
		number is printed after it.  If the parent is a
		member of a cycle, the cycle number is printed between
		the name and the index number.

 If the parents of the function cannot be determined, the word
 `<spontaneous>' is printed in the `name' field, and all the other
 fields are blank.

 For the function's children, the fields have the following meanings:

     self	This is the amount of time that was propagated directly
		from the child into the function.

     children	This is the amount of time that was propagated from the
		child's children to the function.

     called	This is the number of times the function called
		this child `/' the total number of times the child
		was called.  Recursive calls by the child are not
		listed in the number after the `/'.

     name	This is the name of the child.  The child's index
		number is printed after it.  If the child is a
		member of a cycle, the cycle number is printed
		between the name and the index number.

 If there are any cycles (circles) in the call graph, there is an
 entry for the cycle-as-a-whole.  This entry shows who called the
 cycle (as parents) and the members of the cycle (as children.)
 The `+' recursive calls entry shows the number of function calls that
 were internal to the cycle, and the calls entry for each member shows,
 for that member, how many times it was called from other members of
 the cycle.

Copyright (C) 2012-2021 Free Software Foundation, Inc.

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.

Index by function name

  [17] _GLOBAL__sub_I_main     [3] Matrix::sum_abs_rows() [20] std::setprecision(int)
  [18] parse_command_line(int, char**, unsigned int&, unsigned int&, unsigned long&, timeval&, int&) [16] Matrix::sum_abs_matrix() [5] std::abs(float)
  [19] __static_initialization_and_destruction_0(int, int) [14] Matrix::Matrix(int, int) [13] std::setw(int)
   [6] Matrix::print_matrix() [15] Matrix::~Matrix()       [1] main
   [2] Matrix::sum_abs_cols()  [4] Matrix::operator[](int)
