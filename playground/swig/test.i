/* test.i */
%module test
%{
  extern void addInt(int x, int y);
  extern void sumList(PyObject *int_list);
%}

extern void addInt(int x, int y);
extern void sumList(PyObject *int_list);
