#include <pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "../include/nn/networks/network.h"
#include "../include/nn/networks/feedforward_state_value_network.h"
#include "../include/nn/networks/linear_function_approximator.h"
#include "../include/nn/networks/recurrent_state_value_network.h"
#include "../include/nn/networks/expanding_linear_function_approximator.h"
#include "../include/nn/networks/imprinting_wide_network.h"

namespace py = pybind11;

PYBIND11_MODULE(FlexibleNN, m) {
    py::class_<Network>(m, "Network")
        .def(py::init<>())
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
        .def("reset_trace", &ContinuallyAdaptingNetwork::reset_trace);

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
        .def(py::init<float, int, int>())
        .def("print_graph", &ContinuallyAdaptingNetwork::print_graph)
        .def("viz_graph", &ContinuallyAdaptingNetwork::viz_graph)
        .def("set_print_bool", &ContinuallyAdaptingNetwork::set_print_bool)
        .def("get_viz_graph", &ContinuallyAdaptingNetwork::get_viz_graph)
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
        .def(py::init<int, int, int, std::vector<std::pair<float,float>>, float, float, float, bool>())
        .def("step", &ImprintingWideNetwork::step)
        .def("get_feature_bounds", &ImprintingWideNetwork::get_feature_bounds)
        .def("get_feature_utilities", &ImprintingWideNetwork::get_feature_utilities);

}
