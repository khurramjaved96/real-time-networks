//
// Created by taodav on 24/6/21.
//

#ifndef FLEXIBLENN_TEST_ADD_DELETE_H
#define FLEXIBLENN_TEST_ADD_DELETE_H

#include "test.h"

class TestAddDelete : public TestCase {
public:
    TestAddDelete(float step_size, int width, int seed);
    void add_feature(float step_size);
    void delete_feature();
};


#endif //FLEXIBLENN_TEST_ADD_DELETE_H
