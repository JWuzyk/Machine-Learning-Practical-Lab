#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>

#include "HoG.cpp"

extern "C" {
static PyObject *hogpy_hog(PyObject *self, PyObject *args) {
  PyArrayObject *image;
  unsigned long nb_bins;
  double cwidth;
  unsigned long block_size;
  PyObject* unsigned_dirs_obj;
  double clip_val;
  if (!PyArg_ParseTuple(args, "O!kdkOd", &PyArray_Type, &image, &nb_bins,
                        &cwidth, &block_size, &unsigned_dirs_obj, &clip_val)) {
    return nullptr;
  }
  const bool unsigned_dirs = PyObject_IsTrue(unsigned_dirs_obj);

  // your code goes here
}

static PyMethodDef HogpyMethods[] = {
    {"hog", hogpy_hog, METH_VARARGS,
     "Compute the HOG feature vector for an image."},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef hogymodule = {
    PyModuleDef_HEAD_INIT, "hogpy", /* name of module */
    nullptr,                        /* module documentation, may be nullptr */
    -1, /* size of per-interpreter state of the module,
           or -1 if the module keeps state in global variables. */
    HogpyMethods,
    nullptr, /* slots */
    nullptr, /* traverse */
    nullptr, /* clear */
    nullptr, /* free*/
};
}

PyMODINIT_FUNC PyInit_hogpy() {
  import_array();
  return PyModule_Create(&hogymodule);
}
