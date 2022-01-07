The following files are included with this submission:

FinalReport.pdf: Writeup of the final report.

C++ Module:

orderNet.cpp: This implements the C++ class that integrates with the existing TritonRoute C++ implementation to implement OrderNet features. This is primarily responsible for the receiving data from the router and shaping it into JSON messages to send to the Python module. It also implements the sorting based on the respons from the Python module.

orderNet.hpp: Header file for the OrderNet C++ class that is used to include in the rest of the router.


Python Module:

OrderNetEnv.py: Implements the OrderNet custom OpenAI gym environment. This is primarily responsible for parsing messages from the C++ module and providing feedback to the OpenAI gym/Stable Baselines3 training framework. The step() function is where most work happens, including applying the action, calculating the reward, and updating the observation state.

Train.py: Script that controls the training process for the AI agent, utilizing the custom OrderNetEnv environment.

Inference.py: Script that evaluates a benchmark design utilizing OrderNet, based on previously-saved weights.


All code is available at https://github.com/robertviramontes/OrderNet, in particular in src/drt/orderNet.
