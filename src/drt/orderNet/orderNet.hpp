#ifndef __ORDERNET_H__
#define __ORDERNET_H__

#include "Python.h"
#include "db/drObj/drNet.h"
#include <vector>

class OrderNet
{
public:

  OrderNet();
  ~OrderNet();

  void Tester();
  void Train(std::vector<fr::drNet*> ripupNets);

private:
  PyObject *pInstance_;

};

#endif
