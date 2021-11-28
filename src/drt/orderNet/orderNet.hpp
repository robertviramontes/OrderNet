#ifndef __ORDERNET_H__
#define __ORDERNET_H__

#include "Python.h"

class OrderNet
{
public:

  OrderNet();
  ~OrderNet();

  void Tester();

private:
  PyObject *pInstance_;

};

#endif
