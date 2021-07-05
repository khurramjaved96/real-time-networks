#include <vector>
#include <pybind11.h>
#include <pybind11/stl.h>
#include "../include/neural_networks/networks/adaptive_network.h"

namespace py = pybind11;

PYBIND11_MODULE(FlexibleNN, m) {
    py::class_<ContinuallyAdaptingNetwork>(m, "ContinuallyAdaptingNetwork")
        .def(py::init<float, int, int, int>())
        .def("step", &ContinuallyAdaptingNetwork::step)
        .def("set_input_values", &ContinuallyAdaptingNetwork::set_input_values)
        .def("reset_trace", &ContinuallyAdaptingNetwork::reset_trace)
        .def("read_output_values", &ContinuallyAdaptingNetwork::read_output_values)
        .def("read_all_values", &ContinuallyAdaptingNetwork::read_all_values)
        .def("get_input_size", &ContinuallyAdaptingNetwork::get_input_size)
        .def("get_total_synapses", &ContinuallyAdaptingNetwork::get_total_synapses)
        .def("get_viz_graph", &ContinuallyAdaptingNetwork::get_viz_graph)
        .def("add_feature", &ContinuallyAdaptingNetwork::add_feature)
        .def("introduce_targets", py::overload_cast<std::vector<float>, float, float>(&ContinuallyAdaptingNetwork::introduce_targets), "targets, gamma, lambda")
        .def("introduce_targets", py::overload_cast<std::vector<float>, float, float, std::vector<bool>>(&ContinuallyAdaptingNetwork::introduce_targets), "targets, gamma, lambda, no_grad");
}

