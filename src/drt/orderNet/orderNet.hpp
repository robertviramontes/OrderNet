#ifndef __ORDERNET_H__
#define __ORDERNET_H__

#include "Python.h"
#include <vector>

#include "db/drObj/drNet.h"
#include "db/obj/frNet.h"

#include <zmq.hpp>
#include <nlohmann/json.hpp>


using  json = nlohmann::json;

class OrderNet
{
public:

  OrderNet();
  ~OrderNet();

  void Train(std::vector<fr::drNet*>& ripupNets);
  void SendReward(int numViolations, unsigned long long wireLength);

private:
  PyObject *pInstance_;
  zmq::context_t context_;
  zmq::socket_t sender_;
  void sortFromResponse(std::vector<fr::drNet*>& ripupNets, zmq::message_t& reply);
  zmq::message_t jsonInMessage(json& j);
};

#endif
