//
// Created by Khurram Javed on 2021-05-31.
//

#ifndef INCLUDE_NEURAL_NETWORKS_DYNAMIC_ELEM_H_
#define INCLUDE_NEURAL_NETWORKS_DYNAMIC_ELEM_H_


class dynamic_elem {
public:
    int references;

    dynamic_elem();

    virtual ~dynamic_elem() = default;

    void decrement_reference();

    void increment_reference();
};

#endif //INCLUDE_NEURAL_NETWORKS_DYNAMIC_ELEM_H_
