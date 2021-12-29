cmake_minimum_required(VERSION 3.19)
project(NativeModules)

set(CMAKE_CXX_STANDARD 14)

add_library(NativeModules library.cpp library.h)