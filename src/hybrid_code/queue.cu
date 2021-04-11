//
// Created by Khurram Javed on 2021-04-11.
//
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <exception>
#include <stdexcept>

//
struct queue_elem {
    float value;
    queue_elem *next_pointer;
    queue_elem *prev_pointer;

    queue_elem() {
        next_pointer = nullptr;
        prev_pointer = nullptr;
    }
};

class base_queue {
protected:
    queue_elem *back;
    queue_elem *front;
public:
    base_queue() {
        back = nullptr;
        front = nullptr;
    }

    virtual void add_elem(float value) {
    }

    virtual float pop_front() {
        return 0;
    }

    void print_queue() {
        queue_elem *temp = back;
        while (temp != nullptr) {
            std::cout << temp->value << ",";
            temp = temp->next_pointer;
        }
        std::cout << "\n";
    }
};

class device_queue : public base_queue {
public:
    ~device_queue() {
        queue_elem *temp = back;
        while (temp != nullptr) {
            queue_elem *temp_for_free = temp;
            temp = temp->next_pointer;
            cudaFree(temp_for_free);
        }
    }

//
    device_queue() = default;

    void add_elem(float value) override {
        void *test;
        cudaMallocManaged(&test, sizeof(queue_elem));
//        queue_elem *new_elem = new queue_elem();
        auto *new_elem = static_cast<queue_elem *>(test);
        new_elem->next_pointer = nullptr;
        new_elem->value = value;
        if (back == nullptr) {
            if (front != nullptr) {
                std::cout << "Impossible situation \n";
                exit(1);
            }
            new_elem->prev_pointer = nullptr;
            back = new_elem;
            front = new_elem;
        } else {
            new_elem->prev_pointer = nullptr;
            new_elem->next_pointer = back;
            back->prev_pointer = new_elem;
            back = new_elem;
        }
    }

    float pop_front() override {
        if (front == nullptr) {
            std::cout << "Can't pop element from an empty device_queue\n";
            exit(1);
        }
        float return_val = front->value;
        if (front->prev_pointer != nullptr) {
            front->prev_pointer->next_pointer = nullptr;
        }
        queue_elem *temp = front->prev_pointer;
        cudaFree(front);
        front = temp;
        return return_val;
    }
};


class host_queue : public base_queue {
public:
    ~host_queue() {
        queue_elem *temp = back;
        while (temp != nullptr) {
            queue_elem *temp_for_free = temp;
            temp = temp->next_pointer;
            delete temp_for_free;
        }
    }

    host_queue() = default;

    void add_elem(float value) override {
        auto *new_elem = new queue_elem;
        new_elem->next_pointer = nullptr;
        new_elem->value = value;
        if (back == nullptr) {
            if (front != nullptr) {
                std::cout << "Impossible situation \n";
                exit(1);
            }
            new_elem->prev_pointer = nullptr;
            back = new_elem;
            front = new_elem;
        } else {
            new_elem->prev_pointer = nullptr;
            new_elem->next_pointer = back;
            back->prev_pointer = new_elem;
            back = new_elem;
        }
    }

    float pop_front() override {
        if (front == nullptr) {
            std::cout << "Can't pop element from an empty device_queue\n";
            exit(1);
        }
        float return_val = front->value;
        if (front->prev_pointer != nullptr) {
            front->prev_pointer->next_pointer = nullptr;
        }
        queue_elem *temp = front->prev_pointer;
        delete front;
        front = temp;
        return return_val;
    }
};
