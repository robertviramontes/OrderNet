#include "orderNet.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include "odb/geom.h"
#include "dr/FlexDR.h"
#include "dr/FlexMazeTypes.h"

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
  json jFinish;
  jFinish["type"] = "done";
  sendJson(jFinish);
  sender_.disconnect("tcp://localhost:5555");

}

void OrderNet::Train(fr::FlexDRWorker *worker, std::vector<fr::drNet*>& ripupNets, bool willSort) {
  std::cout << "Train" <<std::endl;
  Rect routeBox;

  auto gridGraph = worker->getGridGraph();
  odb::Point lowerRouteBoxPt, upperRouteBoxPt;
  worker->getRouteBox(routeBox);

  json jInferenceData;
  jInferenceData["type"] = "inferenceData";

  std::vector<int> z_used;

  auto contains = [](std::vector<int> vec, int val){
    return std::find(vec.begin(), vec.end(), val) != vec.end();
  };

  json jNets; 
  for (auto net:ripupNets) {
    json jNet;
    auto frNet = net->getFrNet();
    jNet["name"] = frNet->getName();
    json jPins;
    for (auto& pin:net->getPins()) {
      fr::FlexMazeIdx l; fr::FlexMazeIdx h;
      pin->getAPBbox(l, h);
      
      json jPin;
      jPin["l"]["x"] = l.x(); jPin["l"]["y"] = l.y(); jPin["l"]["z"] = l.z();
      int l_layer_num = gridGraph.getLayerNum(l.z());
      if (!contains(z_used, l_layer_num)) z_used.push_back(l_layer_num); 

      jPin["h"]["x"] = h.x(); jPin["h"]["y"] = h.y(); jPin["h"]["z"] = h.z();
      int h_layer_num = gridGraph.getLayerNum(h.z());
      if (!contains(z_used, h_layer_num)) z_used.push_back(h_layer_num); 

      jNet["pins"].push_back(jPin);
    }

    jNets.push_back(jNet); 
  }

  json jRects;
  for (auto z:z_used) {
    lowerRouteBoxPt.setX(routeBox.xMin());
    lowerRouteBoxPt.setY(routeBox.yMin());
    upperRouteBoxPt.setX(routeBox.xMax());
    upperRouteBoxPt.setY(routeBox.yMax());
    
    fr::FlexMazeIdx lowerRouteBoxIdx, upperRouteBoxIdx;
    gridGraph.getMazeIdx(
      lowerRouteBoxIdx,
      lowerRouteBoxPt,
      z);
    gridGraph.getMazeIdx(
      upperRouteBoxIdx,
      upperRouteBoxPt,
      z);
    
    json jRect;
    jRect["xlo"] = lowerRouteBoxIdx.x(); jRect["xhi"] = upperRouteBoxIdx.x();
    jRect["ylo"] = upperRouteBoxIdx.x(); jRect["yhi"] = upperRouteBoxIdx.y();
    jRect["z"] = lowerRouteBoxIdx.z();

    jRects.push_back(jRect);
  }

  jInferenceData["data"]["nets"] = jNets;
  jInferenceData["data"]["routeBoxes"] = jRects;

  auto numLayers = worker->getTech()->getLayers().size();
  jInferenceData["data"]["numLayers"] = numLayers;
  
  sender_.connect ("tcp://localhost:5555");
  zmq::message_t reply;

  sendJson(jInferenceData);
  // Ack back
  sender_.recv (reply, zmq::recv_flags::none);

  // The python application responds with the net ordering
  // If not willSort, Python should be sending an ack that we can ignore.
  if (willSort) {
    json jRequestSort;
    jRequestSort["type"] = "requestSort";
    sendJson(jRequestSort);
    sender_.recv(reply, zmq::recv_flags::none);
    sortFromResponse(ripupNets, reply);
  }

  sender_.disconnect("tcp://localhost:5555");
}

void OrderNet::sortFromResponse(std::vector<fr::drNet*>& ripupNets, zmq::message_t& reply) {
  // Convert the response to json object
  auto response = json::parse(std::string(static_cast<char*>(reply.data()), reply.size()));

  auto responseComp = [response](fr::drNet* const& a, fr::drNet* const& b) {
    // Sort based on the ordering in the response from the model
    // Return true if A is before B, hence list order should be ascending
    return response[a->getFrNet()->getName()] <  response[b->getFrNet()->getName()];
  };
  
  // Actually sort the nets
  sort(ripupNets.begin(), ripupNets.end(), responseComp);
}

void OrderNet::SendReward(int drIter, bool lastInIteration, int numViolations, unsigned long long wireLength) {
  std::cout << "Send Reward" << std::endl;
  sender_.connect ("tcp://localhost:5555");

  json jRewards;
  jRewards["type"] = "reward";
  jRewards["drIter"] = drIter;
  jRewards["lastInIteration"] = lastInIteration;

  json jMetrics;
  jMetrics["numViolations"] = numViolations;
  jMetrics["wireLength"] = wireLength;
  jRewards["data"] = jMetrics;

  sendJson(jRewards);

  // The python application responds with an ack
  zmq::message_t reply;
  sender_.recv (reply, zmq::recv_flags::none);

  sender_.disconnect("tcp://localhost:5555");
}

void OrderNet::sendJson(json& j) {
  std::string jString = j.dump(4);
  zmq::message_t msg (jString.length());
  memcpy (msg.data(), jString.c_str(), strlen(jString.c_str()));
  sender_.send (msg, zmq::send_flags::none);
}

void OrderNet::rectToJson(const Rect *rect, json j) {
  j["xlo"] = rect->xMin(); j["xhi"] = rect->xMax();
  j["ylo"] = rect->yMin(); j["yhi"] = rect->yMax();
}