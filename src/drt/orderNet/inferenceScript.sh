#!/bin/bash

# for i in 1 2 3 4 5
# do
# echo $i
# /bin/python3 /workspaces/OrderNet/src/drt/orderNet/Inference.py mlp_test1_2500steps_noSkipSr.zip > /home/share/mlp/test2.$i.log
# mv /home/share/ispd18_test2.def /home/share/mlp/ispd18_test2.$i.def
# done

# for i in 1 2 3 4 5
# do
# echo $i
# /bin/python3 /workspaces/OrderNet/src/drt/orderNet/Inference.py cnn_test1_2500steps_noSkipSr.zip > /home/share/cnn/test2.$i.log
# mv /home/share/ispd18_test2.def /home/share/cnn/ispd18_test2.$i.def
# done

for i in 1 2 3
do
echo $i
/bin/python3 /workspaces/OrderNet/src/drt/orderNet/Inference.py mlp_test2_5000steps_noSkipSr.zip > /home/share/test2net/mlp/test1.$i.log
ls /home/share/
mv /home/share/ispd18_test1.def /home/share/test2net/mlp/ispd18_test1.$i.def
done

for i in 1 2 3
do
echo $i
/bin/python3 /workspaces/OrderNet/src/drt/orderNet/Inference.py cnn_test2_5000steps_noSkipSr.zip > /home/share/test2net/cnn/test1.$i.log
ls /home/share
mv /home/share/ispd18_test1.def /home/share/test2net/cnn/ispd18_test1.$i.def
done