//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/nn/neuron.h"
#include <assert.h>
#include <iostream>
#include <utility>
#include <algorithm>
#include <vector>
#include "../../include/utils.h"


Neuron::Neuron(bool is_input, bool is_output) {
    value = 0;
    value_before_firing = 0;
    is_recurrent_neuron = false;
    id = neuron_id_generator;
    useless_neuron = false;
    this->average_activation = 0;
    neuron_id_generator++;
    this->is_output_neuron = is_output;
    is_input_neuron = is_input;
    memory_made = 0;
    neuron_age = 0;
    is_mature = false;
    references = 0;
    neuron_utility = 0;
    drinking_age = 159999;
}

/**
 * Fire a neuron. Use the update_value calculated gradient_activation to set this->gradient_activation to
 * the activation by applying an activation function (in this case ReLU) to the calculated gradient_activation.
 * @param time_step: time step that this neuron fires. Used for recording our activation gradient_activation firing time.
 */



void Neuron::fire(int time_step) {
// Temp hack

    if (this->past_activations.size() > 200) {
//        std::cout << "neuron.cpp: Activations acculumating: Unless propagating networks for over 100 layers, this is a memory leak.\n";
//        exit(1);
        this->past_activations.pop();
    }
    if (this->error_gradient.size() > 200) {
//        std::cout << "neuron.cpp: Gradients acculumating: Unless propagating networks for over 100 layers, this is a memory leak.\n";
//        exit(1);
        this->error_gradient.pop();
    }



//  We first set the gradient_activation of the neuron. value_before_firing was set by either inputs being
//  set or by preceding neurons.
    this->old_value = this->value;
    this->value = this->forward(value_before_firing);
    this->shadow_error_prediction = shadow_error_prediction_before_firing;

//  Here we apply our nonlinearity, or our activation function.
//  In this case we stick to ReLU.

//    if (this->is_relu && this->value < 0) {
//        this->value = sigmoid(this->value);
//        this->value = 0;
//        this->average_activation *= 0.99999;
//    } else {
////      Keep a running average of our activations.
//        if (std::abs(this->value) > this->average_activation) {
//            this->average_activation = std::abs(this->value);
//        }
//        this->average_activation = this->average_activation * 0.95 + 0.05 * std::abs(this->value);
//    }
    value_before_firing = 0;
    shadow_error_prediction_before_firing = 0;

//  Record this activation for gradient calculation purposes
//    auto activation_val = std::pair<float, int>(this->gradient_activation, time_step);
    message_activation activation_val;
    activation_val.gradient_activation = this->value;
    activation_val.time = time_step;
    activation_val.error_prediction_value = this->shadow_error_prediction;


    this->past_activations.push(activation_val);



//  Pass this record to our outgoing synapses
    for (auto it : this->outgoing_synapses) {
        message_activation activation_val2;
        activation_val2.gradient_activation = this->value;
        activation_val2.time = time_step;
        activation_val2.error_prediction_value = this->shadow_error_prediction;

        if (it->output_neuron->is_recurrent_neuron) {
//            std::cout << "From To " << it->input_neuron->id << " " << it->output_neuron->id << std::endl;
            if (it->output_neuron->is_output_neuron) {
                std::cout << "Output neuron can't be recurrent\n";
                exit(1);
            }
//            std::cout << "Recurrent output " << it->id << std::endl;
            if (it->get_recurrent_status()) {
                it->TH = this->value + it->output_neuron->recurrent_synapse->weight * it->TH;
//                it->TH = this->gradient_activation;
                if (it->id != it->output_neuron->recurrent_synapse->id) {
                    std::cout << "Incorrect recurrent synapse specification\n";
                    exit(1);
                }
            } else {
                it->TH = this->value + it->output_neuron->recurrent_synapse->weight * it->TH;
//                it->TH = this->gradient_activation;
            }
            activation_val2.gradient_activation = it->TH;
        } else {
//            std::cout << "FASLE From To " << it->input_neuron->id << " " << it->output_neuron->id << std::endl;
        }
        it->weight_assignment_past_activations.push(activation_val2);
    }
//    std::cout << "Fired\n";
}

/**
 * For this neuron, calculate the outgoing gradient_activation (pre activation function) for this time step and set it to
 * value_before_firing.
 * Additionally, when the neuron reaches maturity (age >= 20k), scale the
 * incoming weights so the current node's incoming activation is on average 1, and scale
 * the outgoing weights so that the outgoing activation stays the same.
 */


void Neuron::update_value() {
    this->neuron_age++;
    if (this->neuron_age == this->drinking_age && !this->is_output_neuron) {
        this->is_mature = true;
    }

//  Reset our gradient_activation holder
    this->value_before_firing = 0;
    this->shadow_error_prediction_before_firing = 0;

    if (this->neuron_age == this->drinking_age && !this->is_input_neuron && this->average_activation > 0 &&
        this->outgoing_synapses.size() > 0) {
        float scale = 1 / this->average_activation;
        for (auto it : this->incoming_synapses) {
            if (!it->get_recurrent_status()) {
                it->weight = it->weight * scale;
                it->step_size = 0;
                it->turn_off_idbd();
            }
        }

        if (this->outgoing_synapses.size() == 0 ||
            (this->outgoing_synapses.size() > 1 && !this->is_recurrent_neuron) ||
            (this->outgoing_synapses.size() > 2)) {
            std::cout << "Too many outgoing synapses; shouldn't happen\t" << this->outgoing_synapses.size() << "\n";
            std::cout << "ID\t" << this->neuron_id_generator << " Age \t" << this->neuron_age << std::endl;
            exit(1);
        }
//        this->outgoing_synapses[0]->set_shadow_weight(false);

        for (auto out_g : this->outgoing_synapses) {
//            out_g->weight = out_g->weight * this->average_activation;
            if (!out_g->get_recurrent_status()) {
//                std::cout << "Gets here\n";
//                exit(1);
                out_g->set_shadow_weight(false);
                out_g->weight = 0;
                out_g->step_size = 1e-4;
                out_g->turn_on_idbd();
            }
        }
        this->average_activation = 1;
    }

//  Age our neuron like a fine wine and set the next values of our neuron.
    for (auto &it : this->incoming_synapses) {
        it->age++;
        if (it->in_shadow_mode) {
            this->shadow_error_prediction_before_firing += it->weight * it->input_neuron->value;
        } else {
            this->value_before_firing += it->weight * it->input_neuron->value;
        }
    }
}


bool to_delete_ss(synapse *s) {
    return s->is_useless;
}

/**
 * For each incoming synapse of a neuron, add the gradient from the error in this
 * neuron to its grad_queue for weight assignment. If we do pass gradients backwards,
 * also pass the gradient from the error to grad_queue for use in back propagation.
 */


void Neuron::forward_gradients() {
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
            if (it->in_shadow_mode)
                grad_temp.error = this->error_gradient.front().error_shadow_prediction;
            else
                grad_temp.error = this->error_gradient.front().error;
            grad_temp.distance_travelled = this->error_gradient.front().distance_travelled + 1;
            if (it->propagate_gradients)
                it->grad_queue.push(grad_temp);
            it->grad_queue_weight_assignment.push(grad_temp);
        }  //  Remove this gradient from our list of things needed to pass back
        this->error_gradient.pop();
    }
}

/**
 * NOTE: If you are not VERY familiar with the backprop algorithm, I highly recommend
 * doing some reading before going through this function.
 */
void Neuron::propagate_error() {
    float accumulate_gradient = 0;
    std::vector<int> time_vector;
    std::vector<int> distance_vector;
    std::vector<int> activation_time_required_list;
    std::vector<int> queue_len_vector;
    std::vector<float> error_vector;
    std::vector <message> messages_q;
    int time_check = 99999;

//  No gradient computation required for prediction nodes
    if (!this->outgoing_synapses.empty()) {
        bool flag = false;
        bool wait = false;

//      We look at all outgoing synapses
        for (auto &output_synapses_iterator : this->outgoing_synapses) {
            // Iterate over all outgoing synapses. We want to make sure
//          Skip this if there are no gradients to propagate for this synapse
            if (!output_synapses_iterator->grad_queue.empty()) {
//              This diff in time_step and distance_travelled is essentially "how long until I activate this gradient"
//              Currently, b/c of grad_temp.distance_travelled = error_gradient.front().distance_travelled + 1
//              this means this will always be this->past_activations.front().second - 2.

//              So now we need to match the right past activation with the activation time required.
//              Since we always truncate gradients after 1 step, this corresponds to having a past activation time
//              the same as the time step the gradient was calculated - 2. grad distance_travelled is
//              always 1 in this case.

//              Remove all past activations that are older than the activation time required of the earliest gradient
                while (!output_synapses_iterator->grad_queue.empty() && !this->past_activations.empty() &&
                       this->past_activations.front().time > output_synapses_iterator->grad_queue.front().time_step -
                                                             output_synapses_iterator->grad_queue.front().distance_travelled -
                                                             1) {
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

//                  Check to see if the grad isn't ready to be used.
//                  This is the case where the current grad needs to wait
//                  for other nodes to propagate backwards.
                    if (this->past_activations.front().time < activation_time_required) {
                        wait = true;
                    }
                    if (!wait) {
                        time_vector.push_back(output_synapses_iterator->grad_queue.front().time_step);
                        distance_vector.push_back(output_synapses_iterator->grad_queue.front().distance_travelled);
                        queue_len_vector.push_back(output_synapses_iterator->grad_queue.size());
                        error_vector.push_back(output_synapses_iterator->grad_queue.front().error);
                        messages_q.push_back(output_synapses_iterator->grad_queue.front());

//                      Here we accumulate all our grads wrt the forward node activation
//                      according to the backprop algorithm.
//                      Only accumulate gradient if activation was non-zero.

                        accumulate_gradient += output_synapses_iterator->weight *
                                               output_synapses_iterator->grad_queue.front().gradient *
                                               this->backward(this->past_activations.front().gradient_activation);


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


        if (flag || time_vector.empty())
            return;

//      Remove all the grads we just processed
        for (auto &it : this->outgoing_synapses) {
            if (!it->grad_queue.empty() && !wait) {
                it->grad_queue.pop();
            }
        }

//      check all errors are the same (from the same target)
        float err = error_vector[0];
        for (int a = 0; a < error_vector.size(); a++) {
            if (error_vector[a] != err) {
                std::cout << "Weight = " << this->average_activation << std::endl;
                std::cout << "Neuron.cpp : Shouldn't happen\n";
                std::cout << error_vector[a] << " " << err << std::endl;
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


void Neuron::propagate_deep_error() {
    float accumulate_gradient = 0;
    std::vector<int> time_vector;
    std::vector<int> distance_vector;
    std::vector<int> activation_time_required_list;
    std::vector <message> messages_q;
    int time_check = 99999;

//  No gradient computation required for prediction nodes
    if (!this->outgoing_synapses.empty()) {
        bool flag = false;

//      We look at all outgoing synapses
        for (auto &output_synapses_iterator : this->outgoing_synapses) {
            // Iterate over all outgoing synapses. We want to make sure
//          Skip this if there are no gradients to propagate for this synapse
            if (!output_synapses_iterator->grad_queue.empty()) {
//              This diff in time_step and distance_travelled is essentially "how long until I activate this gradient"
//              Currently, b/c of grad_temp.distance_travelled = error_gradient.front().distance_travelled + 1
//              this means this will always be this->past_activations.front().second - 2.

//              So now we need to match the right past activation with the activation time required.
//              Since we always truncate gradients after 1 step, this corresponds to having a past activation time
//              the same as the time step the gradient was calculated - 2. grad distance_travelled is
//              always 1 in this case.

//              Remove all past activations that are older than the activation time required of the earliest gradient
                int activation_time_required = output_synapses_iterator->grad_queue.front().time_step -
                                               output_synapses_iterator->grad_queue.front().distance_travelled - 1;
                while (!output_synapses_iterator->grad_queue.empty() && !this->past_activations.empty() &&
                       this->past_activations.front().time > activation_time_required) {
                    output_synapses_iterator->grad_queue.pop();
                }

//              This means all the gradients left past here need to be passed back.


//              If we have the situation where an outgoing synapse "skips" neurons
//              This synapse's grad calculation needs to wait until the other chain of neurons
//              is done propagating backwards.
//              grad_queue will be empty in the case that you have a few backprop steps before
//              your corresponding gradient arrives.

                if (output_synapses_iterator->grad_queue.empty()) {
//                  Waiting for gradient from other paths; skipping propagation
                    flag = true;
                }

                if (!flag) {
                    assert(!output_synapses_iterator->grad_queue.empty());
//                  Here we have gradients to process
                    activation_time_required = output_synapses_iterator->grad_queue.front().time_step -
                                               output_synapses_iterator->grad_queue.front().distance_travelled - 1;
                    activation_time_required_list.push_back(activation_time_required);


                    if (this->past_activations.front().time < activation_time_required) {
                        std::cout << "Shouldn't happen in normal operation. Implementation deferred for later\n";
                        exit(1);
                    }


//                  Check to see if the grad isn't ready to be used.
//                  This is the case where the current grad needs to wait
//                  for other nodes to propagate backwards.

                    time_vector.push_back(output_synapses_iterator->grad_queue.front().time_step);
                    distance_vector.push_back(output_synapses_iterator->grad_queue.front().distance_travelled);

//                      Here we accumulate all our grads wrt the forward node activation
//                      according to the backprop algorithm.
//                      Only accumulate gradient if activation was non-zero.

                    accumulate_gradient += output_synapses_iterator->weight *
                                           output_synapses_iterator->grad_queue.front().gradient *
                                           this->backward(this->past_activations.front().gradient_activation);
//                    }

//                      Check that all activaation_time_required are the same
                    if (time_check == 99999) {
                        time_check = activation_time_required;
                    } else {
                        if (time_check != activation_time_required) {
                            flag = true;
                        }
                    }
                }
            } else {
                flag = true;
            }
        }


        if (flag)
            return;

//      Remove all the grads we just processed
        for (auto &it : this->outgoing_synapses) {
            if (!it->grad_queue.empty()) {
                it->grad_queue.pop();
            }
        }


//      Now we make a message to pass our grad of our loss w.r.t. this activation to this neuron
        message n_message(accumulate_gradient, time_vector[0]);
        n_message.error = 1;
        n_message.gamma = 0;
        n_message.lambda = 0;
        auto it = std::max_element(distance_vector.begin(), distance_vector.end());
        n_message.distance_travelled = *it;

//      Remove the activation we just processed
        this->past_activations.pop();
        this->error_gradient.push(n_message);
    }
}
//
/**
 * Mark synapses and neurons for deletion. Synapses will only get deleted if its age is > 70k.
 * Neurons will only be deleted if there are no outgoing synapses (and it's not an output neuron of course!)
 */
void Neuron::mark_useless_weights() {
    for (auto &it : this->outgoing_synapses) {
//      Only delete weights if they're older than 70k steps
        if (it->age > 1000000 || (it->age > (it->output_neuron->drinking_age + 20) && it->step_size < 1e-6)) {
            this->is_mature = true;
//          Don't delete input or output neurons
            if (!(it->input_neuron->is_input_neuron && it->output_neuron->is_output_neuron)) {
//              If the average output of this synapse is small (< 0.01), mark it for deletion
                if (std::abs(it->weight) < 0.01) {
                    it->is_useless = true;
                } else if (it->output_neuron->useless_neuron) {
//                  If the neuron this synapse feeds to is is_useless, also mark it for deletion
                    it->is_useless = true;
                }
            }
        }
    }

//  if this current neuron has no outgoing synapses and is not an output or input neuron,
//  delete it and its incoming synapses.
    if (this->outgoing_synapses.empty() && !this->is_output_neuron && !this->is_input_neuron) {
        this->useless_neuron = true;
        for (auto it : this->incoming_synapses)
            it->is_useless = true;
    }
    if (this->is_input_neuron)
        this->useless_neuron = false;
}

/**
 * Delete outgoing and incoming synapses that were marked earlier as is_useless.
 */
void Neuron::prune_useless_weights() {
    std::for_each(
//            std::execution::seq,
            this->outgoing_synapses.begin(),
            this->outgoing_synapses.end(),
            [&](synapse *s) {
                if (s->is_useless) {
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
                if (s->is_useless) {
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
 * @param target: target gradient_activation to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @return: squared error
 */
float Neuron::introduce_targets(float target, int time_step) {
//

    if (!this->past_activations.empty()) {
//      The activation is the output of our NN.
        float error = target - this->past_activations.front().gradient_activation;
        float error_grad = error;

//      Create our error gradient for this neuron
        message m(error_grad, time_step);
        m.error = 1;
        m.lambda = 0;
        m.gamma = 0;
        this->error_gradient.push(m);
        this->past_activations.pop();
        return error * error;
    }
    return 0;
}

/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target gradient_activation to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @param gamma: discount factor
 * @param lambda: eligibility trace decay parameter
 * @return: squared error
 */
float Neuron::introduce_targets(float target, int time_step, float gamma, float lambda) {
//  Introduce a target to a neuron and calculate its error.
//  In this case, target should be our TD target.

    if (!this->past_activations.empty()) {
//      The activation is the output of our NN.
        float error;
        float error_prediction_error;

        error = target - this->past_activations.front().gradient_activation;
        error_prediction_error = error - this->past_activations.front().error_prediction_value;

        float error_grad = error;


//      Create our error gradient for this neuron
        message m(1, time_step);
        m.lambda = lambda;
        m.gamma = gamma;
        m.error = error_grad;
        m.error_shadow_prediction = error_prediction_error;

        this->error_gradient.push(m);
        this->past_activations.pop();
        return error;
    }
    return 0;
}


/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target gradient_activation to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @param gamma: discount factor
 * @param lambda: eligibility trace decay parameter
 * @param no_grad: whether grad is computed
 * @return: squared error
 */
float Neuron::introduce_targets(float target, int time_step, float gamma, float lambda, bool no_grad) {
//  Introduce a target to a neuron and calculate its error.
//  In this case, target should be our TD target.

    if (!this->past_activations.empty()) {
//      The activation is the output of our NN.
        float error;
        float error_prediction_error;
        if (!no_grad)
            error = target - this->past_activations.front().gradient_activation;
            error_prediction_error = error - this->past_activations.front().error_prediction_value;

        float error_grad = error;


//      Create our error gradient for this neuron
        message m(int(!no_grad), time_step);
        m.lambda = lambda;
        m.gamma = gamma;
        m.error = error_grad;
        m.error_shadow_prediction = error_prediction_error;

        this->error_gradient.push(m);
        this->past_activations.pop();
        return error;
    }
    return 0;
}


void Neuron::update_utility() {
    this->neuron_utility = 0;
    for (auto it : this->outgoing_synapses) {
        this->neuron_utility += it->synapse_utility;
    }
}


float SigmoidNeuron::forward(float temp_value) {

    return sigmoid(temp_value);
}

float SigmoidNeuron::backward(float post_activation) {
    return post_activation * (1 - post_activation);

}

float LinearNeuron::forward(float temp_value) {
    return temp_value;
}

float LinearNeuron::backward(float post_activation) {
    return 1;
}


float ReluNeuron::forward(float temp_value) {
    if (temp_value < 0)
        return 0;
    return temp_value;
}

float ReluNeuron::backward(float post_activation) {
    if (post_activation > 0)
        return 1;
    else
        return 0;

}

float LeakyRelu::forward(float temp_value) {
    if (temp_value < 0)
        return this->negative_slope * temp_value;
    return temp_value;
}

float LeakyRelu::backward(float post_activation) {
    if (post_activation >= 0)
        return 1;
    else
        return this->negative_slope;
}

ReluNeuron::ReluNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {};

SigmoidNeuron::SigmoidNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {};

LeakyRelu::LeakyRelu(bool is_input, bool is_output, float negative_slope) : Neuron(is_input, is_output) {
    this->negative_slope = negative_slope;
};

LinearNeuron::LinearNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {};


int64_t Neuron::neuron_id_generator = 0;
