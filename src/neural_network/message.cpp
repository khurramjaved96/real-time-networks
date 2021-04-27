//
// Created by Khurram Javed on 2021-04-01.
//


#include "../../include/neural_networks/message.h"
#include <stdexcept>



message::message(float m, int t) : time_step(t) {
    this->message_value = m;
    this->time_step = t;
    this->distance_travelled = 0;
}

//void message::increment_distance() {
//    this->distance_travelled++;
//}

//
//
//message operator+(message lhs,        // passing lhs by value helps optimize chained a+b+c
//                   const float& rhs) // otherwise, both parameters may be const references
//{
//    lhs.message_value += rhs; // reuse compound assignment
//    return lhs; // return the result by value (uses move constructor)
//}
//
//message operator-(message lhs,        // passing lhs by value helps optimize chained a+b+c
//                  const float& rhs) // otherwise, both parameters may be const references
//{
//    lhs.message_value -= rhs; // reuse compound assignment
//    return lhs; // return the result by value (uses move constructor)
//}
//
//message operator-(message lhs,        // passing lhs by value helps optimize chained a+b+c
//                  const message& rhs) // otherwise, both parameters may be const references
//{
//    if (lhs.time_step != rhs.time_step)
//        throw std::logic_error("Arithmetic operation on message with different time-steps; likely a bug");
//    lhs.message_value -= rhs.message_value; // reuse compound assignment
//    return lhs; // return the result by value (uses move constructor)
//}
//
//message operator+(message lhs,        // passing lhs by value helps optimize chained a+b+c
//                  const message& rhs) // otherwise, both parameters may be const references
//{
//    if (lhs.time_step != rhs.time_step)
//        throw std::logic_error("Arithmetic operation on message with different time-steps; likely a bug");
//    lhs.message_value += rhs.message_value; // reuse compound assignment
//    return lhs; // return the result by value (uses move constructor)
//}