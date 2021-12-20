//
// Created by Khurram Javed on 2021-05-31.
//

#include "../../include/nn/dynamic_elem.h"

dynamic_elem::dynamic_elem() {
  this->references = 0;
}

void dynamic_elem::increment_reference() {
  this->references += 1;
}

void dynamic_elem::decrement_reference() {
  if (this->references > 0)
    this->references -= 1;
}

void dynamic_elem::increment_reference(int steps){
  this->references+= steps;
}

void dynamic_elem::decrement_reference(int steps) {
  if(this->references > steps)
    this->references -= steps;
  else
    this->references = 0;
}