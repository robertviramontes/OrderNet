import subprocess
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

p = subprocess.Popen(["openroad", "/home/share/ispd_sample2.tcl"], stdout=subprocess.PIPE)

received_message = False
while (not received_message):
  print("Waiting to receive")
  message = socket.recv()
  print(message)

  received_message = True

p.wait()
lines = p.stdout.read().decode("utf-8").split("\n")
for line in lines:
    if "INFO DRT-0199" in line:
        print(line)

