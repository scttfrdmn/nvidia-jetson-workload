/**
 * @file python_bindings.cpp
 * @brief Python bindings for the Weather Simulation.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "weather_sim/weather_sim.hpp"
#include "weather_sim/initial_conditions.hpp"
#include "weather_sim/gpu_adaptability.hpp"
#include "weather_sim/output_manager.hpp"

namespace py = pybind11;

namespace weather_sim {

// Helper function to convert a ScalarField2D to a numpy array
py::array_t<scalar_t> scalarFieldToNumpy(const ScalarField2D& field) {
    auto result = py::array_t<scalar_t>({field.height, field.width});
    auto buffer = result.request();
    scalar_t* ptr = static_cast<scalar_t*>(buffer.ptr);
    
    for (index_t y = 0; y < field.height; ++y) {
        for (index_t x = 0; x < field.width; ++x) {
            ptr[y * field.width + x] = field(x, y);
        }
    }
    
    return result;
}

// Helper function to convert a VectorField2D to two numpy arrays
std::pair<py::array_t<scalar_t>, py::array_t<scalar_t>> vectorFieldToNumpy(const VectorField2D& field) {
    auto u_array = py::array_t<scalar_t>({field.height, field.width});
    auto v_array = py::array_t<scalar_t>({field.height, field.width});
    
    auto u_buffer = u_array.request();
    auto v_buffer = v_array.request();
    
    scalar_t* u_ptr = static_cast<scalar_t*>(u_buffer.ptr);
    scalar_t* v_ptr = static_cast<scalar_t*>(v_buffer.ptr);
    
    for (index_t y = 0; y < field.height; ++y) {
        for (index_t x = 0; x < field.width; ++x) {
            scalar_t u, v;
            field.get(x, y, u, v);
            u_ptr[y * field.width + x] = u;
            v_ptr[y * field.width + x] = v;
        }
    }
    
    return std::make_pair(u_array, v_array);
}

// Helper function to convert a numpy array to a ScalarField2D
void numpyToScalarField(py::array_t<scalar_t, py::array::c_style> array, ScalarField2D& field) {
    auto buffer = array.request();
    
    if (buffer.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }
    
    index_t height = static_cast<index_t>(buffer.shape[0]);
    index_t width = static_cast<index_t>(buffer.shape[1]);
    
    if (width != field.width || height != field.height) {
        throw std::runtime_error("Array dimensions must match field dimensions");
    }
    
    scalar_t* ptr = static_cast<scalar_t*>(buffer.ptr);
    
    for (index_t y = 0; y < height; ++y) {
        for (index_t x = 0; x < width; ++x) {
            field(x, y) = ptr[y * width + x];
        }
    }
}

// Helper function to convert two numpy arrays to a VectorField2D
void numpyToVectorField(
    py::array_t<scalar_t, py::array::c_style> u_array,
    py::array_t<scalar_t, py::array::c_style> v_array,
    VectorField2D& field
) {
    auto u_buffer = u_array.request();
    auto v_buffer = v_array.request();
    
    if (u_buffer.ndim != 2 || v_buffer.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }
    
    index_t u_height = static_cast<index_t>(u_buffer.shape[0]);
    index_t u_width = static_cast<index_t>(u_buffer.shape[1]);
    index_t v_height = static_cast<index_t>(v_buffer.shape[0]);
    index_t v_width = static_cast<index_t>(v_buffer.shape[1]);
    
    if (u_width != field.width || u_height != field.height ||
        v_width != field.width || v_height != field.height) {
        throw std::runtime_error("Array dimensions must match field dimensions");
    }
    
    scalar_t* u_ptr = static_cast<scalar_t*>(u_buffer.ptr);
    scalar_t* v_ptr = static_cast<scalar_t*>(v_buffer.ptr);
    
    for (index_t y = 0; y < field.height; ++y) {
        for (index_t x = 0; x < field.width; ++x) {
            field.set(x, y, u_ptr[y * field.width + x], v_ptr[y * field.width + x]);
        }
    }
}

PYBIND11_MODULE(pyweather_sim, m) {
    m.doc() = "Python bindings for the Weather Simulation workload";
    
    // Enumerations
    py::enum_<SimulationModel>(m, "SimulationModel")
        .value("ShallowWater", SimulationModel::ShallowWater)
        .value("Barotropic", SimulationModel::Barotropic)
        .value("PrimitiveEquations", SimulationModel::PrimitiveEquations)
        .value("General", SimulationModel::General)
        .export_values();
    
    py::enum_<IntegrationMethod>(m, "IntegrationMethod")
        .value("ExplicitEuler", IntegrationMethod::ExplicitEuler)
        .value("RungeKutta2", IntegrationMethod::RungeKutta2)
        .value("RungeKutta4", IntegrationMethod::RungeKutta4)
        .value("AdamsBashforth", IntegrationMethod::AdamsBashforth)
        .value("SemiImplicit", IntegrationMethod::SemiImplicit)
        .export_values();
    
    py::enum_<GridType>(m, "GridType")
        .value("Cartesian", GridType::Cartesian)
        .value("Staggered", GridType::Staggered)
        .value("Icosahedral", GridType::Icosahedral)
        .value("SphericalHarmonic", GridType::SphericalHarmonic)
        .export_values();
    
    py::enum_<BoundaryCondition>(m, "BoundaryCondition")
        .value("Periodic", BoundaryCondition::Periodic)
        .value("Reflective", BoundaryCondition::Reflective)
        .value("Outflow", BoundaryCondition::Outflow)
        .value("Custom", BoundaryCondition::Custom)
        .export_values();
    
    py::enum_<ComputeBackend>(m, "ComputeBackend")
        .value("CUDA", ComputeBackend::CUDA)
        .value("CPU", ComputeBackend::CPU)
        .value("Hybrid", ComputeBackend::Hybrid)
        .value("AdaptiveHybrid", ComputeBackend::AdaptiveHybrid)
        .export_values();
    
    py::enum_<DeviceType>(m, "DeviceType")
        .value("Unknown", DeviceType::Unknown)
        .value("CPU", DeviceType::CPU)
        .value("JetsonOrinNX", DeviceType::JetsonOrinNX)
        .value("T4", DeviceType::T4)
        .value("HighEndGPU", DeviceType::HighEndGPU)
        .value("OtherGPU", DeviceType::OtherGPU)
        .export_values();
    
    py::enum_<OutputFormat>(m, "OutputFormat")
        .value("CSV", OutputFormat::CSV)
        .value("NetCDF", OutputFormat::NetCDF)
        .value("VTK", OutputFormat::VTK)
        .value("PNG", OutputFormat::PNG)
        .value("Custom", OutputFormat::Custom)
        .export_values();
    
    // Structures
    py::class_<SimulationConfig>(m, "SimulationConfig")
        .def(py::init<>())
        .def_readwrite("model", &SimulationConfig::model)
        .def_readwrite("grid_type", &SimulationConfig::grid_type)
        .def_readwrite("integration_method", &SimulationConfig::integration_method)
        .def_readwrite("boundary_condition", &SimulationConfig::boundary_condition)
        .def_readwrite("grid_width", &SimulationConfig::grid_width)
        .def_readwrite("grid_height", &SimulationConfig::grid_height)
        .def_readwrite("num_levels", &SimulationConfig::num_levels)
        .def_readwrite("dx", &SimulationConfig::dx)
        .def_readwrite("dy", &SimulationConfig::dy)
        .def_readwrite("dt", &SimulationConfig::dt)
        .def_readwrite("gravity", &SimulationConfig::gravity)
        .def_readwrite("coriolis_f", &SimulationConfig::coriolis_f)
        .def_readwrite("beta", &SimulationConfig::beta)
        .def_readwrite("viscosity", &SimulationConfig::viscosity)
        .def_readwrite("diffusivity", &SimulationConfig::diffusivity)
        .def_readwrite("compute_backend", &SimulationConfig::compute_backend)
        .def_readwrite("double_precision", &SimulationConfig::double_precision)
        .def_readwrite("device_id", &SimulationConfig::device_id)
        .def_readwrite("num_threads", &SimulationConfig::num_threads)
        .def_readwrite("max_time", &SimulationConfig::max_time)
        .def_readwrite("max_steps", &SimulationConfig::max_steps)
        .def_readwrite("output_interval", &SimulationConfig::output_interval)
        .def_readwrite("output_path", &SimulationConfig::output_path)
        .def_readwrite("random_seed", &SimulationConfig::random_seed);
    
    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_readwrite("total_time_ms", &PerformanceMetrics::total_time_ms)
        .def_readwrite("compute_time_ms", &PerformanceMetrics::compute_time_ms)
        .def_readwrite("memory_transfer_time_ms", &PerformanceMetrics::memory_transfer_time_ms)
        .def_readwrite("io_time_ms", &PerformanceMetrics::io_time_ms)
        .def_readwrite("num_steps", &PerformanceMetrics::num_steps)
        .def("reset", &PerformanceMetrics::reset)
        .def("print", &PerformanceMetrics::print);
    
    py::class_<OutputConfig>(m, "OutputConfig")
        .def(py::init<>())
        .def_readwrite("output_dir", &OutputConfig::output_dir)
        .def_readwrite("prefix", &OutputConfig::prefix)
        .def_readwrite("format", &OutputConfig::format)
        .def_readwrite("output_interval", &OutputConfig::output_interval)
        .def_readwrite("compress", &OutputConfig::compress)
        .def_readwrite("include_diagnostics", &OutputConfig::include_diagnostics)
        .def_readwrite("fields", &OutputConfig::fields);
    
    py::class_<DeviceCapabilities>(m, "DeviceCapabilities")
        .def(py::init<>())
        .def_readonly("device_type", &DeviceCapabilities::device_type)
        .def_readonly("compute_capability_major", &DeviceCapabilities::compute_capability_major)
        .def_readonly("compute_capability_minor", &DeviceCapabilities::compute_capability_minor)
        .def_readonly("cuda_cores", &DeviceCapabilities::cuda_cores)
        .def_readonly("multiprocessors", &DeviceCapabilities::multiprocessors)
        .def_readonly("global_memory", &DeviceCapabilities::global_memory)
        .def_readonly("shared_memory_per_block", &DeviceCapabilities::shared_memory_per_block)
        .def_readonly("max_threads_per_block", &DeviceCapabilities::max_threads_per_block)
        .def_readonly("max_threads_per_multiprocessor", &DeviceCapabilities::max_threads_per_multiprocessor)
        .def_readonly("clock_rate_khz", &DeviceCapabilities::clock_rate_khz)
        .def_readonly("memory_clock_rate_khz", &DeviceCapabilities::memory_clock_rate_khz)
        .def_readonly("memory_bus_width", &DeviceCapabilities::memory_bus_width)
        .def_readonly("compute_power_ratio", &DeviceCapabilities::compute_power_ratio)
        .def_readonly("device_name", &DeviceCapabilities::device_name)
        .def("get_summary", &DeviceCapabilities::getSummary);
    
    // Classes
    py::class_<WeatherGrid>(m, "WeatherGrid")
        .def(py::init<index_t, index_t, index_t>(), 
             py::arg("width"), py::arg("height"), py::arg("num_levels") = 1)
        .def(py::init<const SimulationConfig&>())
        .def("reset", &WeatherGrid::reset)
        .def("get_width", &WeatherGrid::getWidth)
        .def("get_height", &WeatherGrid::getHeight)
        .def("get_num_levels", &WeatherGrid::getNumLevels)
        .def("get_dx", &WeatherGrid::getDx)
        .def("get_dy", &WeatherGrid::getDy)
        .def("set_spacing", &WeatherGrid::setSpacing)
        .def("calculate_diagnostics", &WeatherGrid::calculateDiagnostics)
        .def("get_velocity_field", [](WeatherGrid& self) {
            return vectorFieldToNumpy(self.getVelocityField());
        }, py::return_value_policy::move)
        .def("get_height_field", [](WeatherGrid& self) {
            return scalarFieldToNumpy(self.getHeightField());
        }, py::return_value_policy::move)
        .def("get_pressure_field", [](WeatherGrid& self) {
            return scalarFieldToNumpy(self.getPressureField());
        }, py::return_value_policy::move)
        .def("get_temperature_field", [](WeatherGrid& self) {
            return scalarFieldToNumpy(self.getTemperatureField());
        }, py::return_value_policy::move)
        .def("get_humidity_field", [](WeatherGrid& self) {
            return scalarFieldToNumpy(self.getHumidityField());
        }, py::return_value_policy::move)
        .def("get_vorticity_field", [](WeatherGrid& self) {
            return scalarFieldToNumpy(self.getVorticityField());
        }, py::return_value_policy::move)
        .def("set_velocity_field", [](WeatherGrid& self, py::array_t<scalar_t> u, py::array_t<scalar_t> v) {
            numpyToVectorField(u, v, self.getVelocityField());
        })
        .def("set_height_field", [](WeatherGrid& self, py::array_t<scalar_t> h) {
            numpyToScalarField(h, self.getHeightField());
        })
        .def("set_pressure_field", [](WeatherGrid& self, py::array_t<scalar_t> p) {
            numpyToScalarField(p, self.getPressureField());
        })
        .def("set_temperature_field", [](WeatherGrid& self, py::array_t<scalar_t> t) {
            numpyToScalarField(t, self.getTemperatureField());
        })
        .def("set_humidity_field", [](WeatherGrid& self, py::array_t<scalar_t> q) {
            numpyToScalarField(q, self.getHumidityField());
        });
    
    // InitialCondition classes
    py::class_<InitialCondition, std::shared_ptr<InitialCondition>>(m, "InitialCondition")
        .def("initialize", &InitialCondition::initialize)
        .def("get_name", &InitialCondition::getName);
    
    py::class_<UniformInitialCondition, InitialCondition, std::shared_ptr<UniformInitialCondition>>(m, "UniformInitialCondition")
        .def(py::init<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t, scalar_t>(),
             py::arg("u") = 0.0f, py::arg("v") = 0.0f, py::arg("h") = 10.0f,
             py::arg("p") = 1000.0f, py::arg("t") = 300.0f, py::arg("q") = 0.0f);
    
    py::class_<RandomInitialCondition, InitialCondition, std::shared_ptr<RandomInitialCondition>>(m, "RandomInitialCondition")
        .def(py::init<unsigned int, scalar_t>(),
             py::arg("seed") = 0, py::arg("amplitude") = 1.0f);
    
    py::class_<ZonalFlowInitialCondition, InitialCondition, std::shared_ptr<ZonalFlowInitialCondition>>(m, "ZonalFlowInitialCondition")
        .def(py::init<scalar_t, scalar_t, scalar_t>(),
             py::arg("u_max") = 10.0f, py::arg("h_mean") = 10.0f, py::arg("beta") = 0.1f);
    
    py::class_<VortexInitialCondition, InitialCondition, std::shared_ptr<VortexInitialCondition>>(m, "VortexInitialCondition")
        .def(py::init<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t>(),
             py::arg("x_center") = 0.5f, py::arg("y_center") = 0.5f, py::arg("radius") = 0.1f,
             py::arg("strength") = 10.0f, py::arg("h_mean") = 10.0f);
    
    py::class_<JetStreamInitialCondition, InitialCondition, std::shared_ptr<JetStreamInitialCondition>>(m, "JetStreamInitialCondition")
        .def(py::init<scalar_t, scalar_t, scalar_t, scalar_t>(),
             py::arg("y_center") = 0.5f, py::arg("width") = 0.1f,
             py::arg("strength") = 10.0f, py::arg("h_mean") = 10.0f);
    
    py::class_<BreakingWaveInitialCondition, InitialCondition, std::shared_ptr<BreakingWaveInitialCondition>>(m, "BreakingWaveInitialCondition")
        .def(py::init<scalar_t, scalar_t, scalar_t>(),
             py::arg("amplitude") = 1.0f, py::arg("wavelength") = 0.2f, py::arg("h_mean") = 10.0f);
    
    py::class_<FrontInitialCondition, InitialCondition, std::shared_ptr<FrontInitialCondition>>(m, "FrontInitialCondition")
        .def(py::init<scalar_t, scalar_t, scalar_t, scalar_t>(),
             py::arg("y_position") = 0.5f, py::arg("width") = 0.05f,
             py::arg("temp_difference") = 10.0f, py::arg("wind_shear") = 5.0f);
    
    py::class_<MountainInitialCondition, InitialCondition, std::shared_ptr<MountainInitialCondition>>(m, "MountainInitialCondition")
        .def(py::init<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t>(),
             py::arg("x_center") = 0.3f, py::arg("y_center") = 0.5f, py::arg("radius") = 0.1f,
             py::arg("height") = 1.0f, py::arg("u_base") = 5.0f);
    
    py::class_<AtmosphericProfileInitialCondition, InitialCondition, std::shared_ptr<AtmosphericProfileInitialCondition>>(m, "AtmosphericProfileInitialCondition")
        .def(py::init<const std::string&>(), py::arg("profile_name") = "standard");
    
    // OutputManager (base class only, implementations would be in Python)
    py::class_<OutputManager, std::shared_ptr<OutputManager>>(m, "OutputManager")
        .def("initialize", &OutputManager::initialize)
        .def("write_output", &OutputManager::writeOutput)
        .def("finalize", &OutputManager::finalize);
    
    // WeatherSimulation class
    py::class_<WeatherSimulation>(m, "WeatherSimulation")
        .def(py::init<const SimulationConfig&>())
        .def("set_initial_condition", &WeatherSimulation::setInitialCondition)
        .def("set_output_manager", &WeatherSimulation::setOutputManager)
        .def("initialize", &WeatherSimulation::initialize)
        .def("run", &WeatherSimulation::run)
        .def("run_until", &WeatherSimulation::runUntil)
        .def("step", &WeatherSimulation::step)
        .def("get_current_time", &WeatherSimulation::getCurrentTime)
        .def("get_current_step", &WeatherSimulation::getCurrentStep)
        .def("get_dt", &WeatherSimulation::getDt)
        .def("set_dt", &WeatherSimulation::setDt)
        .def("get_config", &WeatherSimulation::getConfig, py::return_value_policy::reference_internal)
        .def("get_current_grid", &WeatherSimulation::getCurrentGrid, py::return_value_policy::reference_internal)
        .def("get_performance_metrics", &WeatherSimulation::getPerformanceMetrics, py::return_value_policy::reference_internal)
        .def("reset_performance_metrics", &WeatherSimulation::resetPerformanceMetrics);
    
    // Factory and manager classes (singleton patterns)
    m.def("register_all_initial_conditions", &registerAllInitialConditions);
    
    // InitialConditionFactory
    py::class_<InitialConditionFactory>(m, "InitialConditionFactory")
        .def_static("get_instance", &InitialConditionFactory::getInstance, py::return_value_policy::reference)
        .def("create_initial_condition", &InitialConditionFactory::createInitialCondition)
        .def("get_available_initial_conditions", &InitialConditionFactory::getAvailableInitialConditions);
    
    // AdaptiveKernelManager (for GPU capabilities)
    py::class_<AdaptiveKernelManager>(m, "AdaptiveKernelManager")
        .def_static("get_instance", &AdaptiveKernelManager::getInstance, py::return_value_policy::reference)
        .def("initialize", &AdaptiveKernelManager::initialize, py::arg("device_id") = 0)
        .def("is_cuda_available", &AdaptiveKernelManager::isCudaAvailable)
        .def("get_device_capabilities", &AdaptiveKernelManager::getDeviceCapabilities, py::return_value_policy::reference_internal)
        .def("get_gpu_workload_ratio", &AdaptiveKernelManager::getGpuWorkloadRatio)
        .def("determine_optimal_backend", &AdaptiveKernelManager::determineOptimalBackend);
}

} // namespace weather_sim