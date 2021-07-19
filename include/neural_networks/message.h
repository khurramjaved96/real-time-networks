//
// Created by Khurram Javed on 2021-04-01.
//

#ifndef FLEXIBLENN_MESSAGE_H
#define FLEXIBLENN_MESSAGE_H


class message {
public:
    float gradient;
    int time_step;
    int distance_travelled;
    float lambda;
    float gamma;
    float error;
    float error_shadow_prediction;

    message(float m, int t);
};

class message_activation {
public:
    float gradient_activation;
    float error_prediction_value;
    int time;
    float TH;
};


#endif //FLEXIBLENN_MESSAGE_H
