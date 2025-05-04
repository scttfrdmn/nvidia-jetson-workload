/**
 * @file medical_imaging_module.cpp
 * @brief Python bindings for medical imaging workload.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "../cpp/include/medical_imaging/medical_imaging.hpp"
#include "../cpp/include/medical_imaging/gpu_adaptability.hpp"

namespace py = pybind11;
using namespace medical_imaging;

// Helper function to convert numpy array to MedicalImage
MedicalImage numpy_to_medical_image(py::array_t<scalar_t, py::array::c_style> array) {
    // Get array info
    py::buffer_info buffer = array.request();
    
    // Determine dimensionality and size
    std::vector<index_t> size;
    ImageDimension dimension;
    
    if (buffer.ndim == 2) {
        // 2D grayscale image
        size = {static_cast<index_t>(buffer.shape[1]), static_cast<index_t>(buffer.shape[0])};
        dimension = ImageDimension::D2;
    } else if (buffer.ndim == 3) {
        if (buffer.shape[2] <= 4) {
            // 2D color image (RGB or RGBA)
            size = {static_cast<index_t>(buffer.shape[1]), static_cast<index_t>(buffer.shape[0])};
            dimension = ImageDimension::D2;
        } else {
            // 3D grayscale volume
            size = {static_cast<index_t>(buffer.shape[2]), 
                    static_cast<index_t>(buffer.shape[1]), 
                    static_cast<index_t>(buffer.shape[0])};
            dimension = ImageDimension::D3;
        }
    } else if (buffer.ndim == 4) {
        if (buffer.shape[3] <= 4) {
            // 3D color volume (RGB or RGBA)
            size = {static_cast<index_t>(buffer.shape[2]), 
                    static_cast<index_t>(buffer.shape[1]), 
                    static_cast<index_t>(buffer.shape[0])};
            dimension = ImageDimension::D3;
        } else {
            // 4D volume
            size = {static_cast<index_t>(buffer.shape[3]), 
                    static_cast<index_t>(buffer.shape[2]), 
                    static_cast<index_t>(buffer.shape[1]), 
                    static_cast<index_t>(buffer.shape[0])};
            dimension = ImageDimension::D4;
        }
    } else {
        throw std::runtime_error("Unsupported array dimensionality");
    }
    
    // Determine image type and channels
    ImageType type = ImageType::Grayscale;
    index_t channels = 1;
    
    if (buffer.ndim >= 3) {
        if (buffer.shape[buffer.ndim - 1] == 3) {
            type = ImageType::RGB;
            channels = 3;
        } else if (buffer.shape[buffer.ndim - 1] == 4) {
            type = ImageType::RGBA;
            channels = 4;
        } else if (buffer.shape[buffer.ndim - 1] == 2) {
            type = ImageType::Complex;
            channels = 1;
        } else if (buffer.shape[buffer.ndim - 1] > 4) {
            type = ImageType::MultiChannel;
            channels = static_cast<index_t>(buffer.shape[buffer.ndim - 1]);
        }
    }
    
    // Create MedicalImage with the right size
    MedicalImage image(size, dimension, type, channels);
    
    // Copy data
    scalar_t* src_ptr = static_cast<scalar_t*>(buffer.ptr);
    std::vector<scalar_t>& dst_data = image.getDataVector();
    
    // Direct copy for simple cases
    if (buffer.size == static_cast<ssize_t>(dst_data.size())) {
        std::copy(src_ptr, src_ptr + buffer.size, dst_data.data());
    } else {
        // More complex copy for cases with different memory layouts
        // This would need to be implemented based on the specific layouts
        throw std::runtime_error("Unsupported memory layout");
    }
    
    return image;
}

// Helper function to convert MedicalImage to numpy array
py::array_t<scalar_t> medical_image_to_numpy(const MedicalImage& image) {
    // Get image info
    const auto& size = image.getSize();
    ImageDimension dimension = image.getDimension();
    ImageType type = image.getType();
    index_t channels = image.getChannels();
    
    // Determine numpy array shape
    std::vector<ssize_t> shape;
    
    if (dimension == ImageDimension::D2) {
        if (type == ImageType::Grayscale && channels == 1) {
            // 2D grayscale
            shape = {static_cast<ssize_t>(size[1]), static_cast<ssize_t>(size[0])};
        } else {
            // 2D with channels
            shape = {static_cast<ssize_t>(size[1]), static_cast<ssize_t>(size[0]), static_cast<ssize_t>(channels)};
        }
    } else if (dimension == ImageDimension::D3) {
        if (type == ImageType::Grayscale && channels == 1) {
            // 3D volume
            shape = {static_cast<ssize_t>(size[2]), static_cast<ssize_t>(size[1]), static_cast<ssize_t>(size[0])};
        } else {
            // 3D with channels
            shape = {static_cast<ssize_t>(size[2]), static_cast<ssize_t>(size[1]), 
                    static_cast<ssize_t>(size[0]), static_cast<ssize_t>(channels)};
        }
    } else if (dimension == ImageDimension::D4) {
        if (type == ImageType::Grayscale && channels == 1) {
            // 4D volume
            shape = {static_cast<ssize_t>(size[3]), static_cast<ssize_t>(size[2]), 
                    static_cast<ssize_t>(size[1]), static_cast<ssize_t>(size[0])};
        } else {
            // 4D with channels
            shape = {static_cast<ssize_t>(size[3]), static_cast<ssize_t>(size[2]), 
                    static_cast<ssize_t>(size[1]), static_cast<ssize_t>(size[0]), 
                    static_cast<ssize_t>(channels)};
        }
    }
    
    // Create numpy array
    py::array_t<scalar_t> array(shape);
    
    // Copy data
    py::buffer_info buffer = array.request();
    scalar_t* dst_ptr = static_cast<scalar_t*>(buffer.ptr);
    const scalar_t* src_ptr = image.getData();
    
    // Direct copy for simple cases
    if (buffer.size == static_cast<ssize_t>(image.getNumElements())) {
        std::copy(src_ptr, src_ptr + image.getNumElements(), dst_ptr);
    } else {
        // More complex copy for cases with different memory layouts
        // This would need to be implemented based on the specific layouts
        throw std::runtime_error("Unsupported memory layout");
    }
    
    return array;
}

PYBIND11_MODULE(medical_imaging, m) {
    m.doc() = "Medical imaging workload for GPU-accelerated image processing and analysis";
    
    // Enumerations
    py::enum_<ImageDimension>(m, "ImageDimension")
        .value("D2", ImageDimension::D2)
        .value("D3", ImageDimension::D3)
        .value("D4", ImageDimension::D4)
        .export_values();
    
    py::enum_<ImageType>(m, "ImageType")
        .value("Grayscale", ImageType::Grayscale)
        .value("RGB", ImageType::RGB)
        .value("RGBA", ImageType::RGBA)
        .value("Complex", ImageType::Complex)
        .value("MultiChannel", ImageType::MultiChannel)
        .export_values();
    
    py::enum_<ReconstructionMethod>(m, "ReconstructionMethod")
        .value("FilteredBackProjection", ReconstructionMethod::FilteredBackProjection)
        .value("IterativePrimalDual", ReconstructionMethod::IterativePrimalDual)
        .value("CompressedSensing", ReconstructionMethod::CompressedSensing)
        .value("DeepLearning", ReconstructionMethod::DeepLearning)
        .export_values();
    
    py::enum_<FilterMethod>(m, "FilterMethod")
        .value("Gaussian", FilterMethod::Gaussian)
        .value("Median", FilterMethod::Median)
        .value("Bilateral", FilterMethod::Bilateral)
        .value("Anisotropic", FilterMethod::Anisotropic)
        .value("NonLocalMeans", FilterMethod::NonLocalMeans)
        .value("BM3D", FilterMethod::BM3D)
        .value("DeepDenoise", FilterMethod::DeepDenoise)
        .export_values();
    
    py::enum_<SegmentationMethod>(m, "SegmentationMethod")
        .value("Thresholding", SegmentationMethod::Thresholding)
        .value("RegionGrowing", SegmentationMethod::RegionGrowing)
        .value("Watershed", SegmentationMethod::Watershed)
        .value("ActiveContour", SegmentationMethod::ActiveContour)
        .value("GraphCut", SegmentationMethod::GraphCut)
        .value("DeepSegmentation", SegmentationMethod::DeepSegmentation)
        .export_values();
    
    py::enum_<ComputeBackend>(m, "ComputeBackend")
        .value("CUDA", ComputeBackend::CUDA)
        .value("CPU", ComputeBackend::CPU)
        .value("Hybrid", ComputeBackend::Hybrid)
        .value("AdaptiveHybrid", ComputeBackend::AdaptiveHybrid)
        .export_values();
    
    // Configuration struct
    py::class_<ProcessingConfig>(m, "ProcessingConfig")
        .def(py::init<>())
        .def_readwrite("dimension", &ProcessingConfig::dimension)
        .def_readwrite("image_type", &ProcessingConfig::image_type)
        .def_readwrite("image_size", &ProcessingConfig::image_size)
        .def_readwrite("channels", &ProcessingConfig::channels)
        .def_readwrite("compute_backend", &ProcessingConfig::compute_backend)
        .def_readwrite("double_precision", &ProcessingConfig::double_precision)
        .def_readwrite("device_id", &ProcessingConfig::device_id)
        .def_readwrite("num_threads", &ProcessingConfig::num_threads)
        .def_readwrite("scalar_params", &ProcessingConfig::scalar_params)
        .def_readwrite("string_params", &ProcessingConfig::string_params)
        .def_readwrite("vector_params", &ProcessingConfig::vector_params)
        .def_readwrite("save_intermediate", &ProcessingConfig::save_intermediate)
        .def_readwrite("output_path", &ProcessingConfig::output_path)
        .def_readwrite("random_seed", &ProcessingConfig::random_seed);
    
    // Performance metrics struct
    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_readwrite("total_time_ms", &PerformanceMetrics::total_time_ms)
        .def_readwrite("compute_time_ms", &PerformanceMetrics::compute_time_ms)
        .def_readwrite("memory_transfer_time_ms", &PerformanceMetrics::memory_transfer_time_ms)
        .def_readwrite("io_time_ms", &PerformanceMetrics::io_time_ms)
        .def_readwrite("preprocessing_time_ms", &PerformanceMetrics::preprocessing_time_ms)
        .def_readwrite("processing_time_ms", &PerformanceMetrics::processing_time_ms)
        .def_readwrite("postprocessing_time_ms", &PerformanceMetrics::postprocessing_time_ms)
        .def_readwrite("num_iterations", &PerformanceMetrics::num_iterations)
        .def("reset", &PerformanceMetrics::reset)
        .def("print", &PerformanceMetrics::print);
    
    // MedicalImage class
    py::class_<MedicalImage>(m, "MedicalImage")
        .def(py::init<>())
        .def(py::init<const std::vector<index_t>&, ImageDimension, ImageType, index_t>(),
             py::arg("size"), py::arg("dimension") = ImageDimension::D2,
             py::arg("type") = ImageType::Grayscale, py::arg("channels") = 1)
        .def(py::init<const std::string&>())
        .def("load", &MedicalImage::load)
        .def("save", &MedicalImage::save, py::arg("filename"), py::arg("format") = "")
        .def("get_size", &MedicalImage::getSize)
        .def("get_dimension", &MedicalImage::getDimension)
        .def("get_type", &MedicalImage::getType)
        .def("get_channels", &MedicalImage::getChannels)
        .def("get_num_elements", &MedicalImage::getNumElements)
        .def("is_empty", &MedicalImage::isEmpty)
        .def("get_value", &MedicalImage::getValue)
        .def("set_value", &MedicalImage::setValue)
        .def("get_pixel", &MedicalImage::getPixel,
             py::arg("x"), py::arg("y"), py::arg("channel") = 0)
        .def("set_pixel", &MedicalImage::setPixel,
             py::arg("x"), py::arg("y"), py::arg("value"), py::arg("channel") = 0)
        .def("get_voxel", &MedicalImage::getVoxel,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("channel") = 0)
        .def("set_voxel", &MedicalImage::setVoxel,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("value"), py::arg("channel") = 0)
        .def("get_voxel_4d", &MedicalImage::getVoxel4D,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("t"), py::arg("channel") = 0)
        .def("set_voxel_4d", &MedicalImage::setVoxel4D,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("t"), py::arg("value"), py::arg("channel") = 0)
        .def("get_statistics", &MedicalImage::getStatistics, py::arg("channel") = 0)
        .def("normalize", &MedicalImage::normalize, py::arg("channel") = -1)
        .def("resize", &MedicalImage::resize, py::arg("new_size"), py::arg("interpolation") = 1)
        .def("extract_slice", &MedicalImage::extractSlice)
        .def("convert_type", &MedicalImage::convertType)
        .def("apply_window_level", &MedicalImage::applyWindowLevel)
        .def("clone", &MedicalImage::clone);
    
    // NumPy array conversion functions
    m.def("from_numpy", &numpy_to_medical_image, "Convert numpy array to MedicalImage");
    m.def("to_numpy", &medical_image_to_numpy, "Convert MedicalImage to numpy array");
    
    // Image Processor base class
    py::class_<ImageProcessor, std::shared_ptr<ImageProcessor>> processor(m, "ImageProcessor");
    processor.def("initialize", &ImageProcessor::initialize)
        .def("process", &ImageProcessor::process)
        .def("get_config", &ImageProcessor::getConfig)
        .def("set_config", &ImageProcessor::setConfig)
        .def("get_performance_metrics", &ImageProcessor::getPerformanceMetrics)
        .def("reset_performance_metrics", &ImageProcessor::resetPerformanceMetrics)
        .def("set_string_parameter", &ImageProcessor::setStringParameter)
        .def("set_scalar_parameter", &ImageProcessor::setScalarParameter)
        .def("set_vector_parameter", &ImageProcessor::setVectorParameter)
        .def("get_string_parameter", &ImageProcessor::getStringParameter)
        .def("get_scalar_parameter", &ImageProcessor::getScalarParameter)
        .def("get_vector_parameter", &ImageProcessor::getVectorParameter);
    
    // CT Reconstructor
    py::class_<CTReconstructor, ImageProcessor, std::shared_ptr<CTReconstructor>>(m, "CTReconstructor")
        .def(py::init<const ProcessingConfig&>())
        .def("set_method", &CTReconstructor::setMethod)
        .def("get_method", &CTReconstructor::getMethod)
        .def("set_projection_angles", &CTReconstructor::setProjectionAngles)
        .def("get_projection_angles", &CTReconstructor::getProjectionAngles)
        .def("set_num_iterations", &CTReconstructor::setNumIterations)
        .def("get_num_iterations", &CTReconstructor::getNumIterations);
    
    // Image Filter
    py::class_<ImageFilter, ImageProcessor, std::shared_ptr<ImageFilter>>(m, "ImageFilter")
        .def(py::init<const ProcessingConfig&>())
        .def("set_method", &ImageFilter::setMethod)
        .def("get_method", &ImageFilter::getMethod)
        .def("set_filter_parameters", &ImageFilter::setFilterParameters);
    
    // Image Segmenter
    py::class_<ImageSegmenter, ImageProcessor, std::shared_ptr<ImageSegmenter>>(m, "ImageSegmenter")
        .def(py::init<const ProcessingConfig&>())
        .def("set_method", &ImageSegmenter::setMethod)
        .def("get_method", &ImageSegmenter::getMethod)
        .def("set_num_segments", &ImageSegmenter::setNumSegments)
        .def("get_num_segments", &ImageSegmenter::getNumSegments)
        .def("set_seed_points", &ImageSegmenter::setSeedPoints);
    
    // Image Registration
    py::class_<ImageRegistration, ImageProcessor, std::shared_ptr<ImageRegistration>>(m, "ImageRegistration")
        .def(py::init<const ProcessingConfig&>())
        .def("register_images", &ImageRegistration::registerImages)
        .def("get_transform_parameters", &ImageRegistration::getTransformParameters)
        .def("get_transform_matrix", &ImageRegistration::getTransformMatrix);
    
    // Factory for creating processors
    py::class_<ProcessorFactory>(m, "ProcessorFactory")
        .def_static("get_instance", &ProcessorFactory::getInstance, py::return_value_policy::reference)
        .def("create_ct_reconstructor", &ProcessorFactory::createCTReconstructor)
        .def("create_mri_reconstructor", &ProcessorFactory::createMRIReconstructor)
        .def("create_image_filter", &ProcessorFactory::createImageFilter)
        .def("create_image_segmenter", &ProcessorFactory::createImageSegmenter)
        .def("create_image_registration", &ProcessorFactory::createImageRegistration);
    
    // GPU Adaptability components
    
    // Device Type enum
    py::enum_<DeviceType>(m, "DeviceType")
        .value("Unknown", DeviceType::Unknown)
        .value("CPU", DeviceType::CPU)
        .value("JetsonOrinNX", DeviceType::JetsonOrinNX)
        .value("T4", DeviceType::T4)
        .value("HighEndGPU", DeviceType::HighEndGPU)
        .value("OtherGPU", DeviceType::OtherGPU)
        .export_values();
    
    // Device Capabilities struct
    py::class_<DeviceCapabilities>(m, "DeviceCapabilities")
        .def(py::init<>())
        .def_readwrite("device_type", &DeviceCapabilities::device_type)
        .def_readwrite("compute_capability_major", &DeviceCapabilities::compute_capability_major)
        .def_readwrite("compute_capability_minor", &DeviceCapabilities::compute_capability_minor)
        .def_readwrite("cuda_cores", &DeviceCapabilities::cuda_cores)
        .def_readwrite("multiprocessors", &DeviceCapabilities::multiprocessors)
        .def_readwrite("global_memory", &DeviceCapabilities::global_memory)
        .def_readwrite("shared_memory_per_block", &DeviceCapabilities::shared_memory_per_block)
        .def_readwrite("max_threads_per_block", &DeviceCapabilities::max_threads_per_block)
        .def_readwrite("max_threads_per_multiprocessor", &DeviceCapabilities::max_threads_per_multiprocessor)
        .def_readwrite("clock_rate_khz", &DeviceCapabilities::clock_rate_khz)
        .def_readwrite("memory_clock_rate_khz", &DeviceCapabilities::memory_clock_rate_khz)
        .def_readwrite("memory_bus_width", &DeviceCapabilities::memory_bus_width)
        .def_readwrite("compute_power_ratio", &DeviceCapabilities::compute_power_ratio)
        .def_readwrite("device_name", &DeviceCapabilities::device_name)
        .def("get_summary", &DeviceCapabilities::getSummary);
    
    // Kernel Launch Params struct
    py::class_<KernelLaunchParams>(m, "KernelLaunchParams")
        .def(py::init<>())
        .def_readwrite("block_size_x", &KernelLaunchParams::block_size_x)
        .def_readwrite("block_size_y", &KernelLaunchParams::block_size_y)
        .def_readwrite("block_size_z", &KernelLaunchParams::block_size_z)
        .def_readwrite("grid_size_x", &KernelLaunchParams::grid_size_x)
        .def_readwrite("grid_size_y", &KernelLaunchParams::grid_size_y)
        .def_readwrite("grid_size_z", &KernelLaunchParams::grid_size_z)
        .def_readwrite("shared_memory_bytes", &KernelLaunchParams::shared_memory_bytes)
        .def_readwrite("stream_id", &KernelLaunchParams::stream_id)
        .def("compute_grid_dimensions", &KernelLaunchParams::computeGridDimensions);
    
    // Adaptive Kernel Manager
    py::class_<AdaptiveKernelManager>(m, "AdaptiveKernelManager")
        .def_static("get_instance", &AdaptiveKernelManager::getInstance, py::return_value_policy::reference)
        .def("initialize", &AdaptiveKernelManager::initialize, py::arg("device_id") = 0)
        .def("is_cuda_available", &AdaptiveKernelManager::isCudaAvailable)
        .def("get_device_capabilities", &AdaptiveKernelManager::getDeviceCapabilities, py::return_value_policy::reference)
        .def("get_optimal_kernel_params", &AdaptiveKernelManager::getOptimalKernelParams,
             py::arg("operation_name"), py::arg("width"), py::arg("height"), py::arg("depth") = 1)
        .def("get_gpu_workload_ratio", &AdaptiveKernelManager::getGpuWorkloadRatio)
        .def("update_performance_metrics", &AdaptiveKernelManager::updatePerformanceMetrics)
        .def("determine_optimal_backend", &AdaptiveKernelManager::determineOptimalBackend);
    
    // Memory Manager
    py::class_<MemoryManager>(m, "MemoryManager")
        .def_static("get_instance", &MemoryManager::getInstance, py::return_value_policy::reference)
        .def("initialize", &MemoryManager::initialize, py::arg("device_id") = 0)
        .def("synchronize_device", &MemoryManager::synchronizeDevice);
    
    // Hybrid Execution Manager
    py::class_<HybridExecutionManager>(m, "HybridExecutionManager")
        .def_static("get_instance", &HybridExecutionManager::getInstance, py::return_value_policy::reference)
        .def("initialize", &HybridExecutionManager::initialize, 
             py::arg("device_id") = 0, py::arg("num_cpu_threads") = 0)
        .def("set_gpu_workload_ratio", &HybridExecutionManager::setGpuWorkloadRatio)
        .def("get_gpu_workload_ratio", &HybridExecutionManager::getGpuWorkloadRatio)
        .def("adjust_workload_ratio", &HybridExecutionManager::adjustWorkloadRatio);
}