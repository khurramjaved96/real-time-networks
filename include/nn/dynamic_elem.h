//
// Created by Khurram Javed on 2021-05-31.
//

#ifndef INCLUDE_NN_DYNAMIC_ELEM_H_
#define INCLUDE_NN_DYNAMIC_ELEM_H_

class dynamic_elem {
 public:
  int references;

  dynamic_elem();

  virtual ~dynamic_elem() = default;

  void decrement_reference();

  void increment_reference();

  void increment_reference(int);

  void decrement_reference(int);
};

#endif  // INCLUDE_NN_DYNAMIC_ELEM_H_
