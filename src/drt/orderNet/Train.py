import subprocess
import zmq
import os
import json

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

build_dir = os.path.join(os.path.join("/workspaces","OrderNet"), "build")

# subprocess.run(["make", "-j8"], cwd=build_dir)

executable_name = str(os.path.join(build_dir, os.path.join("src","openroad")))

p = subprocess.Popen([executable_name, "/home/share/ispd_sample2.tcl"], stdout=subprocess.PIPE)

received_done = False
while (not received_done):
  message = socket.recv()
  message = message.decode("utf-8")
  if("done" in message):
    received_done = True
    continue
  
  # otherwise, we get data serialized as json 
  net_json = json.loads(message)
  for net in net_json:
    print(net['name'])

  socket.send_string("ack")

p.wait()
lines = p.stdout.read().decode("utf-8").split("\n")
# for line in lines:
#     print(line)

