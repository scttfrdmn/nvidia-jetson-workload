// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "molecular_dynamics/atom.hpp"
#include "molecular_dynamics/molecular_system.hpp"
#include "molecular_dynamics/integrator.hpp"
#include "molecular_dynamics/simulation.hpp"
#include "molecular_dynamics/common.hpp"

namespace py = pybind11;

namespace {

// Helper function to convert Vec3 array to numpy array
py::array_t<double> vec3_array_to_numpy(const std::vector<molecular_dynamics::Vec3>& vec_array) {
    py::array_t<double> result({static_cast<py::ssize_t>(vec_array.size()), 3});
    auto r = result.mutable_unchecked<2>();
    
    for (size_t i = 0; i < vec_array.size(); ++i) {
        r(i, 0) = vec_array[i].x;
        r(i, 1) = vec_array[i].y;
        r(i, 2) = vec_array[i].z;
    }
    
    return result;
}

// Helper function to convert numpy array to Vec3 array
std::vector<molecular_dynamics::Vec3> numpy_to_vec3_array(const py::array_t<double>& numpy_array) {
    auto r = numpy_array.unchecked<2>();
    size_t size = static_cast<size_t>(r.shape(0));
    std::vector<molecular_dynamics::Vec3> result(size);
    
    for (size_t i = 0; i < size; ++i) {
        result[i].x = r(i, 0);
        result[i].y = r(i, 1);
        result[i].z = r(i, 2);
    }
    
    return result;
}

// Helper function to convert atom array to structured numpy array
py::array_t<double> atoms_to_numpy(const std::vector<molecular_dynamics::Atom>& atoms) {
    // Define structured dtype for atoms
    py::array_t<double> positions = py::array_t<double>({static_cast<py::ssize_t>(atoms.size()), 3});
    py::array_t<double> velocities = py::array_t<double>({static_cast<py::ssize_t>(atoms.size()), 3});
    py::array_t<double> forces = py::array_t<double>({static_cast<py::ssize_t>(atoms.size()), 3});
    py::array_t<double> masses = py::array_t<double>({static_cast<py::ssize_t>(atoms.size())});
    py::array_t<double> charges = py::array_t<double>({static_cast<py::ssize_t>(atoms.size())});
    
    auto r_pos = positions.mutable_unchecked<2>();
    auto r_vel = velocities.mutable_unchecked<2>();
    auto r_force = forces.mutable_unchecked<2>();
    auto r_mass = masses.mutable_unchecked<1>();
    auto r_charge = charges.mutable_unchecked<1>();
    
    for (size_t i = 0; i < atoms.size(); ++i) {
        const auto& atom = atoms[i];
        
        // Position
        r_pos(i, 0) = atom.position().x;
        r_pos(i, 1) = atom.position().y;
        r_pos(i, 2) = atom.position().z;
        
        // Velocity
        r_vel(i, 0) = atom.velocity().x;
        r_vel(i, 1) = atom.velocity().y;
        r_vel(i, 2) = atom.velocity().z;
        
        // Force
        r_force(i, 0) = atom.force().x;
        r_force(i, 1) = atom.force().y;
        r_force(i, 2) = atom.force().z;
        
        // Mass and charge
        r_mass(i) = atom.mass();
        r_charge(i) = atom.charge();
    }
    
    // Return as a dictionary
    py::dict result;
    result["positions"] = positions;
    result["velocities"] = velocities;
    result["forces"] = forces;
    result["masses"] = masses;
    result["charges"] = charges;
    
    return result;
}

} // anonymous namespace

PYBIND11_MODULE(pymolecular_dynamics, m) {
    m.doc() = "Python bindings for the Molecular Dynamics simulation";
    
    // Enumerations
    py::enum_<molecular_dynamics::AtomType>(m, "AtomType")
        .value("Hydrogen", molecular_dynamics::AtomType::Hydrogen)
        .value("Carbon", molecular_dynamics::AtomType::Carbon)
        .value("Nitrogen", molecular_dynamics::AtomType::Nitrogen)
        .value("Oxygen", molecular_dynamics::AtomType::Oxygen)
        .value("Sulfur", molecular_dynamics::AtomType::Sulfur)
        .value("Phosphorus", molecular_dynamics::AtomType::Phosphorus)
        .value("Other", molecular_dynamics::AtomType::Other)
        .export_values();
    
    py::enum_<molecular_dynamics::ForceFieldType>(m, "ForceFieldType")
        .value("AMBER", molecular_dynamics::ForceFieldType::AMBER)
        .value("CHARMM", molecular_dynamics::ForceFieldType::CHARMM)
        .value("OPLS", molecular_dynamics::ForceFieldType::OPLS)
        .value("Custom", molecular_dynamics::ForceFieldType::Custom)
        .export_values();
    
    py::enum_<molecular_dynamics::IntegrationType>(m, "IntegrationType")
        .value("VelocityVerlet", molecular_dynamics::IntegrationType::VelocityVerlet)
        .value("Leapfrog", molecular_dynamics::IntegrationType::Leapfrog)
        .value("Beeman", molecular_dynamics::IntegrationType::Beeman)
        .export_values();
    
    py::enum_<molecular_dynamics::ThermostatType>(m, "ThermostatType")
        .value("None", molecular_dynamics::ThermostatType::None)
        .value("Berendsen", molecular_dynamics::ThermostatType::Berendsen)
        .value("Andersen", molecular_dynamics::ThermostatType::Andersen)
        .value("NoseHoover", molecular_dynamics::ThermostatType::NoseHoover)
        .export_values();
    
    py::enum_<molecular_dynamics::GPUDeviceType>(m, "GPUDeviceType")
        .value("Unknown", molecular_dynamics::GPUDeviceType::Unknown)
        .value("CPU", molecular_dynamics::GPUDeviceType::CPU)
        .value("JetsonOrin", molecular_dynamics::GPUDeviceType::JetsonOrin)
        .value("T4", molecular_dynamics::GPUDeviceType::T4)
        .value("HighEnd", molecular_dynamics::GPUDeviceType::HighEnd)
        .export_values();
    
    // Vec3 class
    py::class_<molecular_dynamics::Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("x", &molecular_dynamics::Vec3::x)
        .def_readwrite("y", &molecular_dynamics::Vec3::y)
        .def_readwrite("z", &molecular_dynamics::Vec3::z)
        .def("__add__", [](const molecular_dynamics::Vec3& a, const molecular_dynamics::Vec3& b) { return a + b; })
        .def("__sub__", [](const molecular_dynamics::Vec3& a, const molecular_dynamics::Vec3& b) { return a - b; })
        .def("__mul__", [](const molecular_dynamics::Vec3& a, double b) { return a * b; })
        .def("__rmul__", [](const molecular_dynamics::Vec3& a, double b) { return b * a; })
        .def("__truediv__", [](const molecular_dynamics::Vec3& a, double b) { return a / b; })
        .def("dot", &molecular_dynamics::Vec3::dot)
        .def("cross", &molecular_dynamics::Vec3::cross)
        .def("length", &molecular_dynamics::Vec3::length)
        .def("length_squared", &molecular_dynamics::Vec3::length_squared)
        .def("normalized", &molecular_dynamics::Vec3::normalized)
        .def("normalize", &molecular_dynamics::Vec3::normalize)
        .def("distance", &molecular_dynamics::Vec3::distance)
        .def("distance_squared", &molecular_dynamics::Vec3::distance_squared);
    
    // Atom class
    py::class_<molecular_dynamics::Atom>(m, "Atom")
        .def(py::init<>())
        .def(py::init<const molecular_dynamics::Vec3&, 
                     const molecular_dynamics::Vec3&, 
                     double, 
                     double, 
                     molecular_dynamics::AtomType, 
                     molecular_dynamics::atom_id_t, 
                     molecular_dynamics::res_id_t, 
                     const std::string&, 
                     const std::string&>(),
             py::arg("position"), 
             py::arg("velocity"), 
             py::arg("mass"), 
             py::arg("charge"), 
             py::arg("atom_type"), 
             py::arg("atom_id"), 
             py::arg("residue_id"), 
             py::arg("atom_name") = "", 
             py::arg("residue_name") = "")
        .def("position", &molecular_dynamics::Atom::position)
        .def("set_position", &molecular_dynamics::Atom::set_position)
        .def("velocity", &molecular_dynamics::Atom::velocity)
        .def("set_velocity", &molecular_dynamics::Atom::set_velocity)
        .def("force", &molecular_dynamics::Atom::force)
        .def("set_force", &molecular_dynamics::Atom::set_force)
        .def("mass", &molecular_dynamics::Atom::mass)
        .def("set_mass", &molecular_dynamics::Atom::set_mass)
        .def("charge", &molecular_dynamics::Atom::charge)
        .def("set_charge", &molecular_dynamics::Atom::set_charge)
        .def("type", &molecular_dynamics::Atom::type)
        .def("set_type", &molecular_dynamics::Atom::set_type)
        .def("id", &molecular_dynamics::Atom::id)
        .def("residue_id", &molecular_dynamics::Atom::residue_id)
        .def("atom_name", &molecular_dynamics::Atom::atom_name)
        .def("residue_name", &molecular_dynamics::Atom::residue_name)
        .def("kinetic_energy", &molecular_dynamics::Atom::kinetic_energy)
        .def("update_position", &molecular_dynamics::Atom::update_position)
        .def("update_velocity", &molecular_dynamics::Atom::update_velocity);
    
    // MolecularSystem class
    py::class_<molecular_dynamics::MolecularSystem>(m, "MolecularSystem")
        .def(py::init<>())
        .def(py::init<const std::vector<molecular_dynamics::Atom>&>())
        .def(py::init<const std::vector<molecular_dynamics::Atom>&, 
                     const molecular_dynamics::Vec3&>())
        .def("size", &molecular_dynamics::MolecularSystem::size)
        .def("atom", (const molecular_dynamics::Atom& (molecular_dynamics::MolecularSystem::*)(size_t) const)
                    &molecular_dynamics::MolecularSystem::atom)
        .def("atoms", (const std::vector<molecular_dynamics::Atom>& (molecular_dynamics::MolecularSystem::*)() const)
                     &molecular_dynamics::MolecularSystem::atoms,
             py::return_value_policy::reference_internal)
        .def("add_atom", &molecular_dynamics::MolecularSystem::add_atom)
        .def("box_dimensions", &molecular_dynamics::MolecularSystem::box_dimensions)
        .def("set_box_dimensions", &molecular_dynamics::MolecularSystem::set_box_dimensions)
        .def("has_periodic_boundary", &molecular_dynamics::MolecularSystem::has_periodic_boundary)
        .def("set_periodic_boundary", &molecular_dynamics::MolecularSystem::set_periodic_boundary)
        .def("total_kinetic_energy", &molecular_dynamics::MolecularSystem::total_kinetic_energy)
        .def("total_potential_energy", &molecular_dynamics::MolecularSystem::total_potential_energy)
        .def("total_energy", &molecular_dynamics::MolecularSystem::total_energy)
        .def("temperature", &molecular_dynamics::MolecularSystem::temperature)
        .def("apply_periodic_boundary", &molecular_dynamics::MolecularSystem::apply_periodic_boundary)
        .def("minimum_image_vector", &molecular_dynamics::MolecularSystem::minimum_image_vector)
        .def("calculate_distance_matrix", &molecular_dynamics::MolecularSystem::calculate_distance_matrix)
        .def("update_forces", &molecular_dynamics::MolecularSystem::update_forces)
        .def("update_forces_gpu", &molecular_dynamics::MolecularSystem::update_forces_gpu)
        .def("update_forces_cpu", &molecular_dynamics::MolecularSystem::update_forces_cpu)
        .def("clone", &molecular_dynamics::MolecularSystem::clone)
        .def_static("load_from_pdb", &molecular_dynamics::MolecularSystem::load_from_pdb)
        .def_static("load_with_forcefield", &molecular_dynamics::MolecularSystem::load_with_forcefield,
                  py::arg("pdb_filename"),
                  py::arg("topology_filename"),
                  py::arg("parameter_filename"),
                  py::arg("force_field_type") = molecular_dynamics::ForceFieldType::AMBER)
        .def_static("create_water_box", &molecular_dynamics::MolecularSystem::create_water_box,
                  py::arg("box_size"),
                  py::arg("density") = 1.0)
        .def_static("create_lj_fluid", &molecular_dynamics::MolecularSystem::create_lj_fluid,
                  py::arg("num_particles"),
                  py::arg("box_size"),
                  py::arg("temperature") = molecular_dynamics::DEFAULT_TEMPERATURE,
                  py::arg("seed") = 0);
    
    // Device capabilities
    py::class_<molecular_dynamics::DeviceCapabilities>(m, "DeviceCapabilities")
        .def(py::init<>())
        .def_readwrite("device_type", &molecular_dynamics::DeviceCapabilities::device_type)
        .def_readwrite("compute_capability_major", &molecular_dynamics::DeviceCapabilities::compute_capability_major)
        .def_readwrite("compute_capability_minor", &molecular_dynamics::DeviceCapabilities::compute_capability_minor)
        .def_readwrite("global_memory_bytes", &molecular_dynamics::DeviceCapabilities::global_memory_bytes)
        .def_readwrite("multiprocessor_count", &molecular_dynamics::DeviceCapabilities::multiprocessor_count)
        .def_readwrite("max_threads_per_multiprocessor", &molecular_dynamics::DeviceCapabilities::max_threads_per_multiprocessor)
        .def_readwrite("max_threads_per_block", &molecular_dynamics::DeviceCapabilities::max_threads_per_block)
        .def_readwrite("max_shared_memory_per_block", &molecular_dynamics::DeviceCapabilities::max_shared_memory_per_block)
        .def("get_optimal_block_size", &molecular_dynamics::DeviceCapabilities::get_optimal_block_size)
        .def("get_optimal_tile_size", &molecular_dynamics::DeviceCapabilities::get_optimal_tile_size)
        .def("get_optimal_shared_memory", &molecular_dynamics::DeviceCapabilities::get_optimal_shared_memory)
        .def("get_optimal_grid_size", &molecular_dynamics::DeviceCapabilities::get_optimal_grid_size);
    
    m.def("detect_device_capabilities", &molecular_dynamics::detect_device_capabilities);
    
    // Integrator classes
    py::class_<molecular_dynamics::Integrator>(m, "Integrator")
        .def("initialize", &molecular_dynamics::Integrator::initialize)
        .def("step", &molecular_dynamics::Integrator::step)
        .def("type", &molecular_dynamics::Integrator::type)
        .def("name", &molecular_dynamics::Integrator::name);
    
    py::class_<molecular_dynamics::VelocityVerletIntegrator, molecular_dynamics::Integrator>(m, "VelocityVerletIntegrator")
        .def(py::init<>());
    
    py::class_<molecular_dynamics::LeapfrogIntegrator, molecular_dynamics::Integrator>(m, "LeapfrogIntegrator")
        .def(py::init<>());
    
    py::class_<molecular_dynamics::BeemanIntegrator, molecular_dynamics::Integrator>(m, "BeemanIntegrator")
        .def(py::init<>());
    
    // Thermostat classes
    py::class_<molecular_dynamics::Thermostat>(m, "Thermostat")
        .def("initialize", &molecular_dynamics::Thermostat::initialize)
        .def("apply", &molecular_dynamics::Thermostat::apply)
        .def("type", &molecular_dynamics::Thermostat::type)
        .def("name", &molecular_dynamics::Thermostat::name);
    
    py::class_<molecular_dynamics::BerendsenThermostat, molecular_dynamics::Thermostat>(m, "BerendsenThermostat")
        .def(py::init<double>(), py::arg("coupling_constant") = 0.1);
    
    py::class_<molecular_dynamics::AndersenThermostat, molecular_dynamics::Thermostat>(m, "AndersenThermostat")
        .def(py::init<double>(), py::arg("collision_frequency") = 1.0);
    
    py::class_<molecular_dynamics::NoseHooverThermostat, molecular_dynamics::Thermostat>(m, "NoseHooverThermostat")
        .def(py::init<double>(), py::arg("relaxation_time") = 0.1);
    
    // Simulation class
    py::class_<molecular_dynamics::Simulation>(m, "Simulation")
        .def(py::init<>())
        .def(py::init<std::unique_ptr<molecular_dynamics::MolecularSystem>,
                     std::unique_ptr<molecular_dynamics::Integrator>,
                     std::unique_ptr<molecular_dynamics::Thermostat>,
                     double,
                     double>(),
             py::arg("system"),
             py::arg("integrator") = nullptr,
             py::arg("thermostat") = nullptr,
             py::arg("dt") = molecular_dynamics::DEFAULT_TIMESTEP,
             py::arg("duration") = 10.0)
        .def("current_time", &molecular_dynamics::Simulation::current_time)
        .def("current_step", &molecular_dynamics::Simulation::current_step)
        .def("total_steps", &molecular_dynamics::Simulation::total_steps)
        .def("dt", &molecular_dynamics::Simulation::dt)
        .def("duration", &molecular_dynamics::Simulation::duration)
        .def("temperature", &molecular_dynamics::Simulation::temperature)
        .def("set_temperature", &molecular_dynamics::Simulation::set_temperature)
        .def("system", (const molecular_dynamics::MolecularSystem& (molecular_dynamics::Simulation::*)() const)
                      &molecular_dynamics::Simulation::system,
             py::return_value_policy::reference_internal)
        .def("integrator", &molecular_dynamics::Simulation::integrator, py::return_value_policy::reference_internal)
        .def("thermostat", &molecular_dynamics::Simulation::thermostat, py::return_value_policy::reference_internal)
        .def("set_thermostat", &molecular_dynamics::Simulation::set_thermostat)
        .def("device_capabilities", &molecular_dynamics::Simulation::device_capabilities, py::return_value_policy::reference_internal)
        .def("step", &molecular_dynamics::Simulation::step)
        .def("run", &molecular_dynamics::Simulation::run,
             py::arg("callback") = nullptr,
             py::arg("callback_interval") = 100)
        .def("save_state", &molecular_dynamics::Simulation::save_state)
        .def("load_state", &molecular_dynamics::Simulation::load_state)
        .def("save_trajectory", &molecular_dynamics::Simulation::save_trajectory)
        .def("get_performance_metrics", &molecular_dynamics::Simulation::get_performance_metrics)
        .def("create_visualization_data", &molecular_dynamics::Simulation::create_visualization_data,
             py::arg("include_velocities") = true,
             py::arg("include_forces") = false)
        .def_static("create_water_box_simulation", &molecular_dynamics::Simulation::create_water_box_simulation,
                  py::arg("box_size"),
                  py::arg("integration_type") = molecular_dynamics::IntegrationType::VelocityVerlet,
                  py::arg("thermostat_type") = molecular_dynamics::ThermostatType::Berendsen,
                  py::arg("temperature") = molecular_dynamics::DEFAULT_TEMPERATURE,
                  py::arg("dt") = molecular_dynamics::DEFAULT_TIMESTEP,
                  py::arg("duration") = 10.0)
        .def_static("create_lj_fluid_simulation", &molecular_dynamics::Simulation::create_lj_fluid_simulation,
                  py::arg("num_particles"),
                  py::arg("box_size"),
                  py::arg("integration_type") = molecular_dynamics::IntegrationType::VelocityVerlet,
                  py::arg("thermostat_type") = molecular_dynamics::ThermostatType::Berendsen,
                  py::arg("temperature") = molecular_dynamics::DEFAULT_TEMPERATURE,
                  py::arg("dt") = molecular_dynamics::DEFAULT_TIMESTEP,
                  py::arg("duration") = 10.0,
                  py::arg("seed") = 0)
        .def_static("create_from_pdb", &molecular_dynamics::Simulation::create_from_pdb,
                  py::arg("pdb_filename"),
                  py::arg("integration_type") = molecular_dynamics::IntegrationType::VelocityVerlet,
                  py::arg("thermostat_type") = molecular_dynamics::ThermostatType::Berendsen,
                  py::arg("temperature") = molecular_dynamics::DEFAULT_TEMPERATURE,
                  py::arg("dt") = molecular_dynamics::DEFAULT_TIMESTEP,
                  py::arg("duration") = 10.0);
    
    // Module-level functions
    m.def("vec3_array_to_numpy", &vec3_array_to_numpy);
    m.def("numpy_to_vec3_array", &numpy_to_vec3_array);
    m.def("atoms_to_numpy", &atoms_to_numpy);
    
    // Module constants
    m.attr("DEFAULT_TIMESTEP") = molecular_dynamics::DEFAULT_TIMESTEP;
    m.attr("DEFAULT_TEMPERATURE") = molecular_dynamics::DEFAULT_TEMPERATURE;
    m.attr("DEFAULT_CUTOFF") = molecular_dynamics::DEFAULT_CUTOFF;
    m.attr("BOLTZMANN") = molecular_dynamics::BOLTZMANN;
    m.attr("DEFAULT_BLOCK_SIZE") = molecular_dynamics::DEFAULT_BLOCK_SIZE;
}