//
// Created by Khurram Javed on 2021-04-01.
//

#ifndef FLEXIBLENN_MESSAGE_H
#define FLEXIBLENN_MESSAGE_H


class message{
public:
    float gradient;
    int time_step;
    int distance_travelled;
    float lambda;
    float gamma;
    float error;
    message(float m, int t);
};



#endif //FLEXIBLENN_MESSAGE_H
