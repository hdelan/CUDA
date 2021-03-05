Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls   s/call   s/call  name    
 65.00      2.50     2.50        2     1.25     1.25  transpose_matrix(float*, float*, int, int)
 13.57      3.02     0.52    20002     0.00     0.00  vector_reduction(float const*, int)
  9.92      3.40     0.38                             main
  5.22      3.60     0.20        2     0.10     0.20  abs_value(float*, int, int)
  4.96      3.79     0.19 200000000     0.00     0.00  std::abs(float)
  1.57      3.85     0.06        2     0.03     0.03  std::char_traits<char>::length(char const*)
  0.00      3.85     0.00        2     0.00     0.46  sum_abs_rows(float*, float*, int, int)
  0.00      3.85     0.00        2     0.00     0.00  bool __gnu_cxx::__is_null_pointer<char const>(char const*)
  0.00      3.85     0.00        2     0.00     0.00  int __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)
  0.00      3.85     0.00        2     0.00     0.00  void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*)
  0.00      3.85     0.00        2     0.00     0.00  void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)
  0.00      3.85     0.00        2     0.00     0.00  void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct_aux<char const*>(char const*, char const*, std::__false_type)
  0.00      3.85     0.00        2     0.00     0.03  std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string<std::allocator<char> >(char const*, std::allocator<char> const&)
  0.00      3.85     0.00        2     0.00     0.00  std::__cxx11::stoi(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, unsigned long*, int)
  0.00      3.85     0.00        2     0.00     0.00  std::iterator_traits<char const*>::difference_type std::__distance<char const*>(char const*, char const*, std::random_access_iterator_tag)
  0.00      3.85     0.00        2     0.00     0.00  std::iterator_traits<char const*>::iterator_category std::__iterator_category<char const*>(char const* const&)
  0.00      3.85     0.00        2     0.00     0.00  std::iterator_traits<char const*>::difference_type std::distance<char const*>(char const*, char const*)
  0.00      3.85     0.00        2     0.00     0.00  __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Range_chk::_S_chk(long, std::integral_constant<bool, true>)
  0.00      3.85     0.00        2     0.00     0.00  __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Save_errno::_Save_errno()
  0.00      3.85     0.00        2     0.00     0.00  __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Save_errno::~_Save_errno()
  0.00      3.85     0.00        1     0.00     0.00  _GLOBAL__sub_I_main
  0.00      3.85     0.00        1     0.00     1.70  sum_abs_cols(float*, float*, int, int)
  0.00      3.85     0.00        1     0.00     0.06  parse_command_line(int, char**, unsigned int&, unsigned int&, unsigned long&, timeval&, int&)
  0.00      3.85     0.00        1     0.00     0.00  __static_initialization_and_destruction_0(int, int)
  0.00      3.85     0.00        1     0.00     0.00  std::setprecision(int)

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


granularity: each sample hit covers 2 byte(s) for 0.26% of 3.85 seconds

index % time    self  children    called     name
                                                 <spontaneous>
[1]    100.0    0.38    3.47                 main [1]
                0.00    1.70       1/1           sum_abs_cols(float*, float*, int, int) [3]
                1.25    0.00       1/2           transpose_matrix(float*, float*, int, int) [2]
                0.00    0.46       1/2           sum_abs_rows(float*, float*, int, int) [4]
                0.00    0.06       1/1           parse_command_line(int, char**, unsigned int&, unsigned int&, unsigned long&, timeval&, int&) [10]
                0.00    0.00       2/20002       vector_reduction(float const*, int) [5]
                0.00    0.00       1/1           std::setprecision(int) [31]
-----------------------------------------------
                1.25    0.00       1/2           main [1]
                1.25    0.00       1/2           sum_abs_cols(float*, float*, int, int) [3]
[2]     64.8    2.50    0.00       2         transpose_matrix(float*, float*, int, int) [2]
-----------------------------------------------
                0.00    1.70       1/1           main [1]
[3]     44.3    0.00    1.70       1         sum_abs_cols(float*, float*, int, int) [3]
                1.25    0.00       1/2           transpose_matrix(float*, float*, int, int) [2]
                0.00    0.46       1/2           sum_abs_rows(float*, float*, int, int) [4]
-----------------------------------------------
                0.00    0.46       1/2           main [1]
                0.00    0.46       1/2           sum_abs_cols(float*, float*, int, int) [3]
[4]     23.7    0.00    0.91       2         sum_abs_rows(float*, float*, int, int) [4]
                0.52    0.00   20000/20002       vector_reduction(float const*, int) [5]
                0.20    0.19       2/2           abs_value(float*, int, int) [6]
-----------------------------------------------
                0.00    0.00       2/20002       main [1]
                0.52    0.00   20000/20002       sum_abs_rows(float*, float*, int, int) [4]
[5]     13.5    0.52    0.00   20002         vector_reduction(float const*, int) [5]
-----------------------------------------------
                0.20    0.19       2/2           sum_abs_rows(float*, float*, int, int) [4]
[6]     10.2    0.20    0.19       2         abs_value(float*, int, int) [6]
                0.19    0.00 200000000/200000000     std::abs(float) [7]
-----------------------------------------------
                0.19    0.00 200000000/200000000     abs_value(float*, int, int) [6]
[7]      4.9    0.19    0.00 200000000         std::abs(float) [7]
-----------------------------------------------
                0.06    0.00       2/2           std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string<std::allocator<char> >(char const*, std::allocator<char> const&) [9]
[8]      1.6    0.06    0.00       2         std::char_traits<char>::length(char const*) [8]
-----------------------------------------------
                0.00    0.06       2/2           parse_command_line(int, char**, unsigned int&, unsigned int&, unsigned long&, timeval&, int&) [10]
[9]      1.6    0.00    0.06       2         std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string<std::allocator<char> >(char const*, std::allocator<char> const&) [9]
                0.06    0.00       2/2           std::char_traits<char>::length(char const*) [8]
                0.00    0.00       2/2           void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*) [19]
-----------------------------------------------
                0.00    0.06       1/1           main [1]
[10]     1.6    0.00    0.06       1         parse_command_line(int, char**, unsigned int&, unsigned int&, unsigned long&, timeval&, int&) [10]
                0.00    0.06       2/2           std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string<std::allocator<char> >(char const*, std::allocator<char> const&) [9]
                0.00    0.00       2/2           std::__cxx11::stoi(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, unsigned long*, int) [22]
-----------------------------------------------
                0.00    0.00       2/2           void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag) [20]
[17]     0.0    0.00    0.00       2         bool __gnu_cxx::__is_null_pointer<char const>(char const*) [17]
-----------------------------------------------
                0.00    0.00       2/2           std::__cxx11::stoi(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, unsigned long*, int) [22]
[18]     0.0    0.00    0.00       2         int __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int) [18]
                0.00    0.00       2/2           __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Save_errno::_Save_errno() [27]
                0.00    0.00       2/2           __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Range_chk::_S_chk(long, std::integral_constant<bool, true>) [26]
                0.00    0.00       2/2           __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Save_errno::~_Save_errno() [28]
-----------------------------------------------
                0.00    0.00       2/2           std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string<std::allocator<char> >(char const*, std::allocator<char> const&) [9]
[19]     0.0    0.00    0.00       2         void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*) [19]
                0.00    0.00       2/2           void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct_aux<char const*>(char const*, char const*, std::__false_type) [21]
-----------------------------------------------
                0.00    0.00       2/2           void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct_aux<char const*>(char const*, char const*, std::__false_type) [21]
[20]     0.0    0.00    0.00       2         void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag) [20]
                0.00    0.00       2/2           bool __gnu_cxx::__is_null_pointer<char const>(char const*) [17]
                0.00    0.00       2/2           std::iterator_traits<char const*>::difference_type std::distance<char const*>(char const*, char const*) [25]
-----------------------------------------------
                0.00    0.00       2/2           void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*) [19]
[21]     0.0    0.00    0.00       2         void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct_aux<char const*>(char const*, char const*, std::__false_type) [21]
                0.00    0.00       2/2           void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag) [20]
-----------------------------------------------
                0.00    0.00       2/2           parse_command_line(int, char**, unsigned int&, unsigned int&, unsigned long&, timeval&, int&) [10]
[22]     0.0    0.00    0.00       2         std::__cxx11::stoi(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, unsigned long*, int) [22]
                0.00    0.00       2/2           int __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int) [18]
-----------------------------------------------
                0.00    0.00       2/2           std::iterator_traits<char const*>::difference_type std::distance<char const*>(char const*, char const*) [25]
[23]     0.0    0.00    0.00       2         std::iterator_traits<char const*>::difference_type std::__distance<char const*>(char const*, char const*, std::random_access_iterator_tag) [23]
-----------------------------------------------
                0.00    0.00       2/2           std::iterator_traits<char const*>::difference_type std::distance<char const*>(char const*, char const*) [25]
[24]     0.0    0.00    0.00       2         std::iterator_traits<char const*>::iterator_category std::__iterator_category<char const*>(char const* const&) [24]
-----------------------------------------------
                0.00    0.00       2/2           void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag) [20]
[25]     0.0    0.00    0.00       2         std::iterator_traits<char const*>::difference_type std::distance<char const*>(char const*, char const*) [25]
                0.00    0.00       2/2           std::iterator_traits<char const*>::iterator_category std::__iterator_category<char const*>(char const* const&) [24]
                0.00    0.00       2/2           std::iterator_traits<char const*>::difference_type std::__distance<char const*>(char const*, char const*, std::random_access_iterator_tag) [23]
-----------------------------------------------
                0.00    0.00       2/2           int __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int) [18]
[26]     0.0    0.00    0.00       2         __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Range_chk::_S_chk(long, std::integral_constant<bool, true>) [26]
-----------------------------------------------
                0.00    0.00       2/2           int __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int) [18]
[27]     0.0    0.00    0.00       2         __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Save_errno::_Save_errno() [27]
-----------------------------------------------
                0.00    0.00       2/2           int __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int) [18]
[28]     0.0    0.00    0.00       2         __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Save_errno::~_Save_errno() [28]
-----------------------------------------------
                0.00    0.00       1/1           __libc_csu_init [40]
[29]     0.0    0.00    0.00       1         _GLOBAL__sub_I_main [29]
                0.00    0.00       1/1           __static_initialization_and_destruction_0(int, int) [30]
-----------------------------------------------
                0.00    0.00       1/1           _GLOBAL__sub_I_main [29]
[30]     0.0    0.00    0.00       1         __static_initialization_and_destruction_0(int, int) [30]
-----------------------------------------------
                0.00    0.00       1/1           main [1]
[31]     0.0    0.00    0.00       1         std::setprecision(int) [31]
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

  [29] _GLOBAL__sub_I_main    [18] int __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int) [24] std::iterator_traits<char const*>::iterator_category std::__iterator_category<char const*>(char const* const&)
   [3] sum_abs_cols(float*, float*, int, int) [8] std::char_traits<char>::length(char const*) [7] std::abs(float)
   [4] sum_abs_rows(float*, float*, int, int) [19] void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*) [25] std::iterator_traits<char const*>::difference_type std::distance<char const*>(char const*, char const*)
   [2] transpose_matrix(float*, float*, int, int) [20] void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag) [26] __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Range_chk::_S_chk(long, std::integral_constant<bool, true>)
   [5] vector_reduction(float const*, int) [21] void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct_aux<char const*>(char const*, char const*, std::__false_type) [27] __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Save_errno::_Save_errno()
  [10] parse_command_line(int, char**, unsigned int&, unsigned int&, unsigned long&, timeval&, int&) [9] std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string<std::allocator<char> >(char const*, std::allocator<char> const&) [28] __gnu_cxx::__stoa<long, int, char, int>(long (*)(char const*, char**, int), char const*, char const*, unsigned long*, int)::_Save_errno::~_Save_errno()
  [30] __static_initialization_and_destruction_0(int, int) [22] std::__cxx11::stoi(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, unsigned long*, int) [1] main
   [6] abs_value(float*, int, int) [23] std::iterator_traits<char const*>::difference_type std::__distance<char const*>(char const*, char const*, std::random_access_iterator_tag)
  [17] bool __gnu_cxx::__is_null_pointer<char const>(char const*) [31] std::setprecision(int)
