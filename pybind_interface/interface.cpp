#include <pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "../include/nn/networks/network.h"
#include "../include/nn/networks/feedforward_state_value_network.h"
#include "../include/nn/networks/linear_function_approximator.h"
#include "../include/nn/networks/recurrent_state_value_network.h"
#include "../include/nn/networks/expanding_linear_function_approximator.h"
#include "../include/nn/networks/imprinting_wide_network.h"
#include "../include/nn/networks/imprinting_atari_network.h"
#include "../include/nn/synapse.h"
#include "../include/experiment/Metric.h"

namespace py = pybind11;

PYBIND11_MODULE(FlexibleNN, m) {
    py::class_<Network>(m, "Network")
        .def(py::init<>())
        .def_readonly("output_neurons", &Network::output_neurons)
        .def_readonly("all_synapses", &Network::all_synapses)
        .def_readonly("output_synapses", &Network::output_synapses)
        .def_readonly("all_neurons", &Network::all_neurons)
        .def_readonly("input_neurons", &Network::input_neurons)
        .def("get_timestep", &Network::get_timestep)
        .def("set_input_values", &Network::set_input_values)
        .def("step", &Network::step)
        .def("read_output_values", &Network::read_output_values)
        .def("read_all_values", &Network::read_all_values)
//        .def("introduce_targets", py::overload_cast<std::vector<float>>(&Network::introduce_targets), "targets")
        .def("introduce_targets", py::overload_cast<std::vector<float>, float, float>(&Network::introduce_targets), "targets, gamma, lambda")
        .def("introduce_targets", py::overload_cast<float, float, float, std::vector<bool>>(&Network::introduce_targets), "targets, gamma, lambda, no_grad")
        .def("forward_pass_without_side_effects", &Network::forward_pass_without_side_effects)
        .def("get_input_size", &Network::get_input_size)
        .def("get_total_synapses", &Network::get_total_synapses)
        .def("get_total_neurons", &Network::get_total_neurons)
        .def("reset_trace", &Network::reset_trace)
        .def("collect_garbage", &Network::collect_garbage)
        .def("print_graph", &Network::print_graph)
        .def("viz_graph", &Network::viz_graph)
        .def("get_viz_graph", &Network::get_viz_graph);

//        py::class_<LinearFunctionApproximator>(m, "LinearFunctionApproximator")
//        .def(py::init<int, int, float, float, bool>())
//        .def("get_timestep", &LinearFunctionApproximator::get_timestep)
//        .def("set_input_values", &LinearFunctionApproximator::set_input_values)
//        .def("step", &LinearFunctionApproximator::step)
//        .def("read_output_values", &LinearFunctionApproximator::read_output_values)
//        .def("read_all_values", &LinearFunctionApproximator::read_all_values)
//        //        .def("introduce_targets", py::overload_cast<std::vector<float>>(&Network::introduce_targets), "targets")
//        .def("introduce_targets", py::overload_cast<std::vector<float>, float, float>(&LinearFunctionApproximator::introduce_targets), "targets, gamma, lambda")
//        .def("get_input_size", &LinearFunctionApproximator::get_input_size)
//        .def("get_total_synapses", &LinearFunctionApproximator::get_total_synapses)
//        .def("get_total_neurons", &LinearFunctionApproximator::get_total_neurons)
//        .def("reset_trace", &LinearFunctionApproximator::reset_trace);


    py::class_<ContinuallyAdaptingNetwork, Network>(m, "ContinuallyAdaptingNetwork")
        .def(py::init<float, int, int, float>())
        .def("set_print_bool", &ContinuallyAdaptingNetwork::set_print_bool)
        .def("introduce_targets", py::overload_cast<std::vector<float>>(&ContinuallyAdaptingNetwork::introduce_targets), "targets")
        .def("introduce_targets", py::overload_cast<std::vector<float>, float, float>(&ContinuallyAdaptingNetwork::introduce_targets), "targets, gamma, lambda")
        .def("introduce_targets", py::overload_cast<std::vector<float>, float, float, std::vector<bool>>(&ContinuallyAdaptingNetwork::introduce_targets), "targets, gamma, lambda, no_grad")
        .def("add_feature", &ContinuallyAdaptingNetwork::add_feature);


    py::class_<LinearFunctionApproximator, Network>(m, "LinearFunctionApproximator")
        .def(py::init<int, int, float, float, bool>())
        .def("step", &LinearFunctionApproximator::step);

    py::class_<ExpandingLinearFunctionApproximator, Network>(m, "ExpandingLinearFunctionApproximator")
        .def(py::init<int, int, int, float, float, bool>())
        .def("set_input_values", &ExpandingLinearFunctionApproximator::set_input_values)
        .def("step", &ExpandingLinearFunctionApproximator::step);

    py::class_<ImprintingWideNetwork, Network>(m, "ImprintingWideNetwork")
        .def(py::init<int, int, int, std::vector<std::pair<float,float>>, float, float, float, float, bool, int, bool>())
        .def_readonly("bounded_neurons", &ImprintingWideNetwork::bounded_neurons)
        .def("step", &ImprintingWideNetwork::step)
        .def("get_reassigned_bounded_neurons", &ImprintingWideNetwork::get_reassigned_bounded_neurons)
        .def("count_active_bounded_units", &ImprintingWideNetwork::count_active_bounded_units)
        .def("replace_lowest_utility_bounded_unit", &ImprintingWideNetwork::replace_lowest_utility_bounded_unit)
        .def("get_feature_bounds", &ImprintingWideNetwork::get_feature_bounds)
        .def("get_feature_utilities", &ImprintingWideNetwork::get_feature_utilities);

    py::class_<ImprintingAtariNetwork, Network>(m, "ImprintingAtariNetwork")
        .def(py::init<int, int, int, float, float, bool, int, bool, int, int, int, float, bool, bool>())
        .def_readonly("imprinted_features", &ImprintingAtariNetwork::imprinted_features)
        .def("imprint_randomly", &ImprintingAtariNetwork::imprint_randomly)
        .def("imprint_using_optical_flow", &ImprintingAtariNetwork::imprint_using_optical_flow)
        .def("imprint_using_optical_flow_old", &ImprintingAtariNetwork::imprint_using_optical_flow_old)
        .def("set_input_values", &ImprintingAtariNetwork::set_input_values)
        .def("step", &ImprintingAtariNetwork::step);

    py::class_<Metric>(m, "Metric")
        .def(py::init<std::string, std::string, std::vector<std::string>, std::vector<std::string>, std::vector<std::string>>())
        .def("add_value", &Metric::add_value)
        .def("add_values", &Metric::add_values);

    py::class_<Database>(m, "Database")
        .def(py::init<>())
        .def("create_database", &Database::create_database);

    py::class_<synapse>(m, "synapse")
        .def_readonly("id", &synapse::id)
        .def_readonly("is_useless", &synapse::is_useless)
        .def_readonly("age", &synapse::age)
        .def_readonly("weight", &synapse::weight)
        .def_readonly("credit", &synapse::credit)
        .def_readonly("trace", &synapse::trace)
        .def_readonly("meta_step_size", &synapse::meta_step_size)
        .def_readonly("utility_to_keep", &synapse::utility_to_keep)
        .def_readonly("synapse_utility", &synapse::synapse_utility)
        .def_readonly("input_neuron", &synapse::input_neuron)
        .def_readonly("output_neuron", &synapse::output_neuron);


    py::class_<Neuron>(m, "Neuron")
        .def_readonly("id", &Neuron::id)
        .def_readonly("useless_neuron", &Neuron::useless_neuron)
        .def_readonly("neuron_age", &Neuron::neuron_age)
        .def_readonly("is_input_neuron", &Neuron::is_input_neuron)
        .def_readonly("is_output_neuron", &Neuron::is_output_neuron)
        .def_readonly("value", &Neuron::value)
        .def_readonly("average_activation", &Neuron::average_activation)
        .def_readonly("neuron_utility", &Neuron::neuron_utility)
        .def_readonly("sum_of_utility_traces", &Neuron::sum_of_utility_traces)
        .def_readonly("incoming_synapses", &Neuron::incoming_synapses)
        .def_readonly("outgoing_synapses", &Neuron::outgoing_synapses);

    py::class_<BoundedNeuron, Neuron>(m, "BoundedNeuron")
        .def_readonly("num_times_reassigned", &BoundedNeuron::num_times_reassigned);
}
