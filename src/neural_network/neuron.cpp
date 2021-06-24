//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/neural_networks/neuron.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include <vector>
#include "../../include/utils.h"
#include <assert.h>

neuron::neuron(bool activation) {
    value = 0;
    temp_value = 0;
    id = neuron_id;
    neuron_id++;
    useless_neuron = false;
    this->average_activation = 0;
    this->is_output_neuron = false;
    this->activation_type = activation;
    is_input_neuron = false;
    memory_made = 0;
    neuron_age = 0;
    mature = false;
    references = 0;
}

neuron::neuron(bool activation, bool output_n) {
    value = 0;
    temp_value = 0;
    id = neuron_id;
    useless_neuron = false;
    this->average_activation = 0;
    neuron_id++;
    this->is_output_neuron = output_n;
    this->activation_type = activation;
    is_input_neuron = false;
    memory_made = 0;
    neuron_age = 0;
    mature = false;
    references = 0;
}

neuron::neuron(bool activation, bool output_n, int id) {
    value = 0;
    temp_value = 0;
    this->id = id;
    useless_neuron = false;
    this->average_activation = 0;
    neuron_id++;
    this->is_output_neuron = output_n;
    this->activation_type = activation;
    is_input_neuron = false;
    memory_made = 0;
    neuron_age = 0;
    mature = false;
    references = 0;
}

neuron::neuron(bool activation, bool output_n, bool input_n) {
    value = 0;
    temp_value = 0;
    id = neuron_id;
    useless_neuron = false;
    this->average_activation = 0;
    neuron_id++;
    this->is_output_neuron = false;
    this->activation_type = true;
    is_input_neuron = input_n;
    memory_made = 0;
    neuron_age = 0;
    mature = false;
    references = 0;

}

/**
 * Fire a neuron. Use the update_value calculated value to set this->value to
 * the activation by applying an activation function (in this case ReLU) to the calculated value.
 * @param time_step: time step that this neuron fires. Used for recording our activation value firing time.
 */
void neuron::fire(int time_step) {
//   Temp hack
    if (this->past_activations.size() > 50) {
        this->past_activations.pop();

    }
    if (this->error_gradient.size() > 50) {
        this->error_gradient.pop();
    }

//  We first set the value of the neuron. temp_value was set by either inputs being
//  set or by preceding neurons.
    this->value = temp_value;

//  Here we apply our nonlinearity, or our activation function.
//  In this case we stick to ReLU.
    if (this->activation_type && this->value <= 0) {
        this->value = 0;
    } else {
//      Keep a running average of our activations.
        this->average_activation = this->average_activation * 0.95 + 0.05 * std::abs(this->value);
    }
    temp_value = 0;

//  Record this activation for gradient calculation purposes
    auto activation_val = std::pair<float, int>(this->value, time_step);
    this->past_activations.push(activation_val);

//  Pass this record to our outgoing synapses
    for (auto it: this->outgoing_synapses)
        it->weight_assignment_past_activations.push(activation_val);
}

/**
 * For this neuron, calculate the outgoing value (pre activation function) for this time step and set it to
 * temp_value.
 * Additionally, when the neuron reaches maturity (age >= 20k), scale the
 * incoming weights so the current node's incoming activation is on average 1, and scale
 * the outgoing weights so that the outgoing activation stays the same.
 */
void neuron::update_value() {
//  If our neuron hasn't been pruned in 20k steps, it's mature and stays.
    if (this->neuron_age == 19999 and !this->is_output_neuron) {
        this->mature = true;
    }
    this->neuron_age++;
    if (memory_made > 0)
        memory_made--;

//  Reset our value holder
    this->temp_value = 0;

//  If our neuron reaches maturity, we "flip" our activations and scale things
//  so that our average neuron output is 1.
    if (this->neuron_age == 19999 and !this->is_input_neuron and this->average_activation > 0 and
        this->outgoing_synapses.size() > 0) {
        float scale = 1 / this->average_activation;
        for (auto it : this->incoming_synapses) {
            it->weight = it->weight * scale;
        }

        if (this->outgoing_synapses.size() == 0) {
            std::cout << "Too many outgoing synapses; shouldn't happen\t" << this->outgoing_synapses.size() << "\n";
            std::cout << "ID\t" << this->neuron_id << " Age \t" << this->neuron_age << std::endl;
            exit(1);
        }
        for (auto out_g : this->outgoing_synapses) {
            out_g->weight = out_g->weight * this->average_activation;
            out_g->step_size = 1e-4;
            out_g->turn_on_idbd();
        }
        this->average_activation = 1;
    }

//  Age our neuron like a fine wine and set the next values of our neuron.
    for (auto &it : this->incoming_synapses) {
        it->age++;
        this->temp_value += it->weight * it->input_neuron->value;
    }
}

/**
 * For each incoming synapse of a neuron, add the gradient from the error in this
 * neuron to its grad_queue for weight assignment. If we do pass gradients backwards,
 * also pass the gradient from the error to grad_queue for use in back propagation.
 */
void neuron::forward_gradients() {
//  If this neuron has gradients to pass back
    if (!this->error_gradient.empty()) {

//      We do so to all incoming synapses
        for (auto &it : this->incoming_synapses) {

            float message_value;

            message_value = this->error_gradient.front().gradient;

//          We pack our gradient into a new message and pass it back to our incoming synapse.
            message grad_temp(message_value, this->error_gradient.front().time_step);
            grad_temp.lambda = this->error_gradient.front().lambda;
            grad_temp.gamma = this->error_gradient.front().gamma;
            grad_temp.error = this->error_gradient.front().error;
            grad_temp.distance_travelled = this->error_gradient.front().distance_travelled + 1;
            if (it->pass_gradients)
                it->grad_queue.push(grad_temp);
            it->grad_queue_weight_assignment.push(grad_temp);


        }
//      Remove this gradient from our list of things needed to pass back
        this->error_gradient.pop();
    }
}

/**
 * NOTE: If you are not VERY familiar with the backprop algorithm, I highly recommend
 * doing some reading before going through this function.
 */
void neuron::propagate_error() {

    float accumulate_gradient = 0;
    std::vector<int> time_vector;
    std::vector<int> distance_vector;
    std::vector<int> activation_time_required_list;
    std::vector<int> queue_len_vector;
    std::vector<float> error_vector;
    std::vector<message> messages_q;
    int time_check = 99999;

//  No gradient computation required for prediction nodes
    if (!this->outgoing_synapses.empty()) {
        bool flag = false;
        bool wait = false;

//      We look at all outgoing synapses
        for (auto &output_synapses_iterator : this->outgoing_synapses) { //Iterate over all outgoing synapses. We want to make sure

//          Skip this if there are no gradients to propagate for this synapse
            if (!output_synapses_iterator->grad_queue.empty()) {

//              This diff in time_step and distance_travelled is essentially "how long until I activate this gradient"
//              Currently, b/c of grad_temp.distance_travelled = error_gradient.front().distance_travelled + 1
//              this means this will always be this->past_activations.front().second - 2.

//              So now we need to match the right past activation with the activation time required.
//              Since we always truncate gradients after 1 step, this corresponds to having a past activation time
//              the same as the time step the gradient was calculated - 2. grad distance_travelled is always 1 in this case.

//              Remove all past activations that are older than the activation time required of the earliest gradient
                while (!output_synapses_iterator->grad_queue.empty() and !this->past_activations.empty() and
                       this->past_activations.front().second > output_synapses_iterator->grad_queue.front().time_step -
                                                               output_synapses_iterator->grad_queue.front().distance_travelled - 1) {

                    output_synapses_iterator->grad_queue.pop();
                }

//              This means all the gradients left past here need to be passed back.

//              We check to see if we have any past activations
                if (this->past_activations.empty())
                    return;

//              If we have the situation where an outgoing synapse "skips" neurons
//              This synapse's grad calculation needs to wait until the other chain of neurons
//              is done propagating backwards.
//              grad_queue will be empty in the case that you have a few backprop steps before
//              your corresponding gradient arrives.
                bool temp_flag = true;
                if (output_synapses_iterator->grad_queue.empty()) {
//                  Waiting for gradient from other paths; skipping propagation
                    temp_flag = false;
                }

                if (temp_flag) {
                    assert(!output_synapses_iterator->grad_queue.empty());
//                  Here we have gradients to process
                    int activation_time_required = output_synapses_iterator->grad_queue.front().time_step -
                                                   output_synapses_iterator->grad_queue.front().distance_travelled - 1;
                    activation_time_required_list.push_back(activation_time_required);

//                  Check to see if the grad isn't ready to be used. This is the case where the current grad needs to wait
//                  for other nodes to propagate backwards.
                    if (this->past_activations.front().second < activation_time_required) {
                        wait = true;
                    }
                    if (!wait) {
                        time_vector.push_back(output_synapses_iterator->grad_queue.front().time_step);
                        distance_vector.push_back(output_synapses_iterator->grad_queue.front().distance_travelled);
                        queue_len_vector.push_back(output_synapses_iterator->grad_queue.size());
                        error_vector.push_back(output_synapses_iterator->grad_queue.front().error);
                        messages_q.push_back(output_synapses_iterator->grad_queue.front());

//                      Here we accumulate all our grads wrt the forward node activation according to the backprop algorithm.
//                      Only accumulate gradient if activation was non-zero.
                        if (this->past_activations.front().first > 0 or !this->activation_type) {
//                            std::cout << "Past activation = " << this->past_activations.front().first << std::endl;
                            accumulate_gradient += output_synapses_iterator->weight *
                                                   output_synapses_iterator->grad_queue.front().gradient;
                        }

//                      Check that all activaation_time_required are the same
                        if (time_check == 99999) {
                            time_check = activation_time_required;
                        } else {
                            if (time_check != activation_time_required) {

                                flag = true;
                            }
                        }
                    }
                }
            }

        }


        if (flag or time_vector.empty())
            return;

//      Remove all the grads we just processed
        for (auto &it: this->outgoing_synapses) {
            if (!it->grad_queue.empty() and !wait) {
                it->grad_queue.pop();
            }
        }

//      check all errors are the same (from the same target)
        float err = error_vector[0];
        for (int a = 0; a < error_vector.size(); a++) {
            if (error_vector[a] != err) {
                std::cout << "Weight = " << this->average_activation << std::endl;
                std::cout << "Neuron.cpp : Shouldn't happen\n";
                exit(1);
            }
        }

//      Now we make a message to pass our grad of our loss w.r.t. this activation to this neuron
        message n_message(accumulate_gradient, time_vector[0]);
        n_message.error = error_vector[0];
        n_message.gamma = messages_q[0].gamma;
        n_message.lambda = messages_q[0].lambda;
        auto it = std::max_element(distance_vector.begin(), distance_vector.end());
        n_message.distance_travelled = *it;

//      Remove the activation we just processed
        this->past_activations.pop();
        this->error_gradient.push(n_message);

    }
}

/**
 * Mark synapses and neurons for deletion. Synapses will only get deleted if its age is > 70k.
 * Neurons will only be deleted if there are no outgoing synapses (and it's not an output neuron of course!)
 */
void neuron::mark_useless_weights() {
    for (auto &it : this->outgoing_synapses) {
//      Only delete weights if they're older than 70k steps
        if (it->age > 69999) {
//          Don't delete input or output neurons
            if (!(it->input_neuron->is_input_neuron and it->output_neuron->is_output_neuron)) {
//              If the average output of this synapse is small (< 0.01), mark it for deletion
                if (this->average_activation * std::abs(it->weight) < 0.01) {
                    it->useless = true;
                } else if (it->output_neuron->useless_neuron) {
//                  If the neuron this synapse feeds to is useless, also mark it for deletion
                    it->useless = true;
                }
            }
        }
    }

//  if this current neuron has no outgoing synapses and is not an output or input neuron,
//  delete it and its incoming synapses.
    if (this->outgoing_synapses.empty() and !this->is_output_neuron and !this->is_input_neuron) {
        this->useless_neuron = true;
        for (auto it : this->incoming_synapses)
            it->useless = true;
    }
}

bool to_delete_ss(synapse *s) {
    return s->useless;
}

/**
 * Delete outgoing and incoming synapses that were marked earlier as useless.
 */
void neuron::prune_useless_weights() {
    std::for_each(
//            std::execution::seq,
            this->outgoing_synapses.begin(),
            this->outgoing_synapses.end(),
            [&](synapse *s) {
                if (s->useless) {
                    s->decrement_reference();
                    if (s->input_neuron != nullptr) {
                        s->input_neuron->decrement_reference();
                        s->input_neuron = nullptr;
                    }
                    if (s->output_neuron != nullptr) {
                        s->output_neuron->decrement_reference();
                        s->output_neuron = nullptr;
                    }
                }
            });

    auto it = std::remove_if(this->outgoing_synapses.begin(), this->outgoing_synapses.end(), to_delete_ss);
    this->outgoing_synapses.erase(it, this->outgoing_synapses.end());

    std::for_each(
//            std::execution::seq,
            this->incoming_synapses.begin(),
            this->incoming_synapses.end(),
            [&](synapse *s) {
                if (s->useless) {
                    s->decrement_reference();
                    if (s->input_neuron != nullptr) {
                        s->input_neuron->decrement_reference();
                        s->input_neuron = nullptr;
                    }
                    if (s->output_neuron != nullptr) {
                        s->output_neuron->decrement_reference();
                        s->output_neuron = nullptr;
                    }
                }
            });
    it = std::remove_if(this->incoming_synapses.begin(), this->incoming_synapses.end(), to_delete_ss);
    this->incoming_synapses.erase(it, this->incoming_synapses.end());

}

/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target value to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 *
 * @return: squared error
 */
float neuron::introduce_targets(float target, int time_step) {
//

    if (!this->past_activations.empty()) {
//      The activation is the output of our NN.
        float error = target - this->past_activations.front().first;
        float error_grad = error;

//      If activation leq zero and we apply relu on this output, don't let gradients flow back
        if (this->past_activations.front().first <= 0 and this->activation_type) {
            error_grad = 0;
        }

//      Create our error gradient for this neuron
        message m(1, time_step);
        m.error = error_grad;
        this->error_gradient.push(m);
        this->past_activations.pop();
        return error * error;
    }
    return 0;
}

/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target value to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @param gamma: discount factor
 * @param lambda: eligibility trace decay parameter
 * @return: squared error
 */
float neuron::introduce_targets(float target, int time_step, float gamma, float lambda) {
//  Introduce a target to a neuron and calculate its error.
//  In this case, target should be our TD target.

    if (!this->past_activations.empty()) {
//      The activation is the output of our NN.

        float error = target - this->past_activations.front().first;
        float error_grad = error;

//      If activation leq zero and we apply relu on this output, don't let gradients flow back
        if (this->past_activations.front().first <= 0 and this->activation_type) {
            std::cout << "Should never get here\n";
            exit(1);
            error_grad = 0;
        }

//      Create our error gradient for this neuron
        message m(1, time_step);
        m.lambda = lambda;
        m.gamma = gamma;
        m.error = error_grad;

        this->error_gradient.push(m);
        this->past_activations.pop();
        return error * error;
    }
    return 0;
}

int neuron::neuron_id = 0;
normal_random neuron::normal_dist = normal_random(0, 1, 0.1);