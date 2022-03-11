cmake_minimum_required(VERSION 3.10)

# set the project name
project(OrderNet)

# add the executable
add_library(OrderNet
            STATIC 
            orderNet.cpp)

SET(PYTHON_COMPILE_FLAGS -I/usr/include/python3.6m -I/usr/include/python3.6m  -Wno-unused-result -Wsign-compare -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector-strong --param=ssp-buffer-size=4 -grecord-gcc-switches   -m64 -mtune=generic -D_GNU_SOURCE -fPIC -fwrapv   -DNDEBUG -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector-strong --param=ssp-buffer-size=4 -grecord-gcc-switches   -m64 -mtune=generic -D_GNU_SOURCE -fPIC )

find_package(cppzmq)

find_package(nlohmann_json 3.2.0 REQUIRED)

target_link_libraries(OrderNet PUBLIC drt PRIVATE cppzmq nlohmann_json::nlohmann_json)
target_include_directories(OrderNet PUBLIC "../src" "../../odb/include")
