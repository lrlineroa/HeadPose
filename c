#!/bin/bash
rm ./e
clear
g++ c.cpp -o e -O2 `pkg-config --cflags --libs opencv`
./e < in
