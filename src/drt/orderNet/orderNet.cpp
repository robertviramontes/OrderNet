#include "orderNet.hpp"

#include <iostream>

#define PMODULE_NAME "OrderNet"
#define PCLASS_NAME "OrderNet"

OrderNet::OrderNet()
{
  std::cout << "Into OrderNet!" << std::endl;
  PyObject *pName, *pModule, *pFunc, *pClass;
  PyObject *pArgs, *pValue;
  int i;

  Py_Initialize();
  pName =PyUnicode_FromString(PMODULE_NAME);
  /* Error checking of pName left out */

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL) {
    pClass = PyObject_GetAttrString(pModule, PCLASS_NAME);

    // Create an instance of the class
    if (PyCallable_Check(pClass)) {
      pArgs = Py_BuildValue("()");
      pInstance_ = PyObject_CallObject(pClass, pArgs);

      Py_DECREF(pClass);
      Py_DECREF(pArgs);
    } else {
      // TODO handle error couldn't find the class
    }

    Py_DECREF(pModule);
  } else {
    std::cerr << "Did not find the module." << std::endl;
  }
}

OrderNet::~OrderNet() {
  if (pInstance_ != NULL) {
    Py_DECREF(pInstance_);
  }
}

void OrderNet::Tester() {
  if (pInstance_ == NULL) {
    std::cerr << "Instance lost." << std::endl;
    return;
  }
  PyObject_CallMethod(pInstance_, "train", NULL);
  PyObject_CallMethod(pInstance_, "inference", NULL);

}