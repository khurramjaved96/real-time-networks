//
// Created by Khurram Javed on 2021-05-31.
//

#ifndef FLEXIBLENN_DYNAMIC_ELEM_H
#define FLEXIBLENN_DYNAMIC_ELEM_H


class dynamic_elem{
public:
    int references;
    dynamic_elem();
    virtual ~dynamic_elem()= default;
    void decrement_reference();
    void increment_reference();
};

#endif //FLEXIBLENN_DYNAMIC_ELEM_H
