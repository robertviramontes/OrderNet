#ifndef __ORDERNET_H__
#define __ORDERNET_H__

#include "Python.h"
#include "db/drObj/drNet.h"
#include "db/obj/frNet.h"
#include <zmq.hpp>
#include <vector>

class OrderNet
{
public:

  OrderNet();
  ~OrderNet();

  void Train(std::vector<fr::drNet*> ripupNets);

private:
  PyObject *pInstance_;
  zmq::context_t context_;
  zmq::socket_t sender_;
};

#endif
