#include "orderNet.hpp"

#include <iostream>
#include <sstream>

#define PMODULE_NAME "OrderNet"
#define PCLASS_NAME "OrderNet"

OrderNet::OrderNet() :
    context_(1),
    sender_(context_, zmq::socket_type::req)
{
  std::cout << "Into OrderNet!" << std::endl;
  PyObject *pName, *pModule, *pClass;
  PyObject *pArgs;

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

  // Tell listeners we are done.
  sender_.connect ("tcp://localhost:5555");
  zmq::message_t request (4);
  memcpy (request.data (), "done", 4);
  sender_.send (request, zmq::send_flags::none);
  sender_.disconnect("tcp://localhost:5555");

}

void OrderNet::Train(std::vector<fr::drNet*>& ripupNets) {
  sender_.connect ("tcp://localhost:5555");

  json jInferenceData;
  jInferenceData["type"] = "inferenceData";
  
  json jNets;
  for (auto net:ripupNets) {
    json jNet;
    auto frNet = net->getFrNet();
    jNet["name"] = frNet->getName();
    jNet["numPinsIn"] = net->getNumPinsIn();

    jNets.push_back(jNet); 
  }

  jInferenceData["data"]["nets"] = jNets;


  auto msg = jsonInMessage(jInferenceData);
  sender_.send (msg, zmq::send_flags::none);

  // The python application responds with the net ordering
  zmq::message_t reply;
  sender_.recv (reply, zmq::recv_flags::none);

  sortFromResponse(ripupNets, reply);

  for (auto net:ripupNets) {
    std::cerr << net->getFrNet()->getName() << std::endl;
  }


  sender_.disconnect("tcp://localhost:5555");
}

void OrderNet::sortFromResponse(std::vector<fr::drNet*>& ripupNets, zmq::message_t& reply) {
  // Convert the response to json object
  auto response = json::parse(static_cast<char*>(reply.data()));

  auto responseComp = [response](fr::drNet* const& a, fr::drNet* const& b) {
    // Sort based on the ordering in the response from the model
    // Return true if A is before B, hence list order should be ascending
    return response[a->getFrNet()->getName()] <  response[b->getFrNet()->getName()];
  };
  
  // Actually sort the nets
  sort(ripupNets.begin(), ripupNets.end(), responseComp);
}

void OrderNet::SendReward(int numViolations, unsigned long long wireLength) {
  sender_.connect ("tcp://localhost:5555");

  json jRewards;
  jRewards["type"] = "reward";

  json jMetrics;
  jMetrics["numViolations"] = numViolations;
  jMetrics["wireLength"] = wireLength;
  jRewards["data"] = jMetrics;

  auto msg = jsonInMessage(jRewards);
  sender_.send (msg, zmq::send_flags::none);

  // The python application responds with the net ordering
  zmq::message_t reply;
  sender_.recv (reply, zmq::recv_flags::none);

  // sortFromResponse(ripupNets, reply);

  // for (auto net:ripupNets) {
  //   std::cerr << net->getFrNet()->getName() << std::endl;
  // }


  sender_.disconnect("tcp://localhost:5555");
}

zmq::message_t OrderNet::jsonInMessage(json& j) {
  std::string jString = j.dump(4);
  zmq::message_t msg (jString.length());
  memcpy (msg.data(), jString.c_str(), strlen(jString.c_str()));
  return msg;
}