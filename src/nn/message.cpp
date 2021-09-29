//
// Created by Khurram Javed on 2021-04-01.
//


#include "../../include/nn/message.h"
#include <stdexcept>

message::message(float m, int t) : time_step(t) {
  this->gradient = m;
  this->time_step = t;
  this->distance_travelled = 0;
  this->lambda = 0;
  this->gamma = 0;
  this->error = 0;
}

message::message(){
  this->gradient = 0;
  this->time_step = 0;
  this->distance_travelled = 0;
  this->lambda = 0;
  this->gamma = 0;
  this->error = 0;
}