#include "orderNet.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

#define PMODULE_NAME "OrderNet"
#define PCLASS_NAME "OrderNet"

using  json = nlohmann::json;

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

  std::cerr << "Ending applcation." << std::endl;
  // Tell listeners we are done.
  sender_.connect ("tcp://localhost:5555");
  zmq::message_t request (4);
  memcpy (request.data (), "done", 4);
  sender_.send (request, zmq::send_flags::none);
  sender_.disconnect("tcp://localhost:5555");

}

void OrderNet::Train(std::vector<fr::drNet*> ripupNets) {
  sender_.connect ("tcp://localhost:5555");

  json jNets;
  for (auto net:ripupNets) {
    json jNet;
    auto frNet = net->getFrNet();
    jNet["name"] = frNet->getName();
    jNet["numPinsIn"] = net->getNumPinsIn();

    jNets.push_back(jNet); 
  }

  std::string message = jNets.dump(4);
  zmq::message_t msg (message.length());
  memcpy (msg.data (), message.c_str(), strlen(message.c_str()));
  sender_.send (msg, zmq::send_flags::none);

  zmq::message_t reply;
  sender_.recv (reply, zmq::recv_flags::none);

  sender_.disconnect("tcp://localhost:5555");
}