#ifndef __ORDERNET_H__
#define __ORDERNET_H__

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <zmq.hpp>

#include "db/drObj/drNet.h"
#include "db/obj/frNet.h"

#define USE_ORDERNET 1

using json = nlohmann::json;

namespace fr {
class FlexDRWorker;
}

class OrderNet
{
 public:
  OrderNet();
  ~OrderNet();

  void Train(fr::FlexDRWorker* worker,
             std::vector<fr::drNet*>& ripupNets,
             bool willSort);
  void SendReward(int drIter,
                  bool lastInIteration,
                  int numViolations,
                  unsigned long long wireLength);

 private:
  zmq::context_t context_;
  zmq::socket_t sender_;
  void sortFromResponse(std::vector<fr::drNet*>& ripupNets,
                        zmq::message_t& reply);
  void sendJson(json& j);
  void rectToJson(const Rect* rect, json j);
  std::string connectionString_;
};

#endif
