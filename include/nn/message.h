//
// Created by Khurram Javed on 2021-04-01.
//

#ifndef INCLUDE_NN_MESSAGE_H_
#define INCLUDE_NN_MESSAGE_H_

class message {
 public:
  float gradient;
  bool remove = false;
  int time_step;
  int distance_travelled;
  float lambda;
  float gamma;
  float error;
  float target;
  float error_shadow_prediction;

  message(float m, int t);
  message();
};

class message_activation {
 public:
  float gradient_activation;
  float value_at_activation;
  float error_prediction_value;
  int time;
  float TH;
};

#endif  // INCLUDE_NN_MESSAGE_H_
