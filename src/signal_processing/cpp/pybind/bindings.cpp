// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <vector>
#include <complex>

#include "signal_processing/wavelet_transform.h"
#include "signal_processing/digital_filtering.h"
#include "signal_processing/spectral_analysis.h"
#include "signal_processing/time_frequency.h"

namespace py = pybind11;

// Convert std::vector<float> to numpy array
py::array_t<float> vector_to_numpy(const std::vector<float>& vec) {
    py::array_t<float> array(vec.size());
    auto buf = array.request();
    float* ptr = static_cast<float*>(buf.ptr);
    std::memcpy(ptr, vec.data(), vec.size() * sizeof(float));
    return array;
}

// Convert numpy array to std::vector<float>
std::vector<float> numpy_to_vector(const py::array_t<float>& array) {
    auto buf = array.request();
    float* ptr = static_cast<float*>(buf.ptr);
    return std::vector<float>(ptr, ptr + buf.size);
}

// Convert WaveletTransformResult to Python dictionary
py::dict transform_result_to_dict(const signal_processing::WaveletTransformResult& result) {
    py::dict dict;
    
    // Convert approximation coefficients
    py::list approx_list;
    for (const auto& coeff : result.approximation_coefficients) {
        approx_list.append(vector_to_numpy(coeff));
    }
    dict["approximation"] = approx_list;
    
    // Convert detail coefficients
    py::list detail_list;
    for (const auto& coeff : result.detail_coefficients) {
        detail_list.append(vector_to_numpy(coeff));
    }
    dict["detail"] = detail_list;
    
    return dict;
}

// Convert Python dictionary to WaveletTransformResult
signal_processing::WaveletTransformResult dict_to_transform_result(const py::dict& dict) {
    signal_processing::WaveletTransformResult result;
    
    // Convert approximation coefficients
    py::list approx_list = dict["approximation"].cast<py::list>();
    result.approximation_coefficients.resize(approx_list.size());
    for (size_t i = 0; i < approx_list.size(); i++) {
        result.approximation_coefficients[i] = numpy_to_vector(approx_list[i].cast<py::array_t<float>>());
    }
    
    // Convert detail coefficients
    py::list detail_list = dict["detail"].cast<py::list>();
    result.detail_coefficients.resize(detail_list.size());
    for (size_t i = 0; i < detail_list.size(); i++) {
        result.detail_coefficients[i] = numpy_to_vector(detail_list[i].cast<py::array_t<float>>());
    }
    
    return result;
}

// Convert WaveletPacketResult to Python dictionary
py::dict packet_result_to_dict(const signal_processing::WaveletPacketResult& result) {
    py::dict dict;
    
    // Convert coefficients
    py::dict coeffs_dict;
    for (size_t level = 0; level < result.coefficients.size(); level++) {
        for (size_t node = 0; node < result.coefficients[level].size(); node++) {
            // Create tuple key for level, node
            auto key = py::make_tuple(level, node);
            coeffs_dict[key] = vector_to_numpy(result.coefficients[level][node]);
        }
    }
    dict["coefficients"] = coeffs_dict;
    dict["levels"] = result.coefficients.size() - 1;
    
    return dict;
}

// Convert Python dictionary to WaveletPacketResult
signal_processing::WaveletPacketResult dict_to_packet_result(const py::dict& dict) {
    signal_processing::WaveletPacketResult result;
    
    // Get levels
    int levels = dict["levels"].cast<int>();
    result.coefficients.resize(levels + 1);
    
    // Convert coefficients
    py::dict coeffs_dict = dict["coefficients"].cast<py::dict>();
    for (auto item : coeffs_dict) {
        auto key = item.first.cast<py::tuple>();
        auto value = item.second.cast<py::array_t<float>>();
        
        int level = key[0].cast<int>();
        int node = key[1].cast<int>();
        
        // Make sure level vector is big enough
        if (level >= result.coefficients.size()) {
            result.coefficients.resize(level + 1);
        }
        
        // Make sure node vector is big enough
        if (node >= result.coefficients[level].size()) {
            result.coefficients[level].resize(node + 1);
        }
        
        // Convert numpy array to vector
        result.coefficients[level][node] = numpy_to_vector(value);
    }
    
    return result;
}

// Convert 2D vector of complex to numpy array
py::array_t<std::complex<float>> complex_vector_to_numpy(const std::vector<std::vector<std::complex<float>>>& vec) {
    if (vec.empty()) {
        return py::array_t<std::complex<float>>(0);
    }
    
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    
    py::array_t<std::complex<float>> array({rows, cols});
    auto buf = array.request();
    std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
    
    for (size_t i = 0; i < rows; i++) {
        std::memcpy(ptr + i * cols, vec[i].data(), cols * sizeof(std::complex<float>));
    }
    
    return array;
}

// Convert numpy array to 2D vector of complex
std::vector<std::vector<std::complex<float>>> numpy_to_complex_vector(const py::array_t<std::complex<float>>& array) {
    auto buf = array.request();
    std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Array must have 2 dimensions");
    }
    
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    
    std::vector<std::vector<std::complex<float>>> result(rows);
    for (size_t i = 0; i < rows; i++) {
        result[i].resize(cols);
        std::memcpy(result[i].data(), ptr + i * cols, cols * sizeof(std::complex<float>));
    }
    
    return result;
}

#ifdef WITH_CUDA
// Check if CUDA is available
bool has_cuda() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount > 0;
}
#endif

PYBIND11_MODULE(_signal_processing, m) {
    m.doc() = "C++ backend for the signal_processing package";
    
    // Version information
    m.attr("__version__") = "0.1.0";
    
#ifdef WITH_CUDA
    // CUDA availability
    m.def("has_cuda", &has_cuda, "Check if CUDA is available");
#endif

    // Wavelet Transform Enums
    py::enum_<signal_processing::WaveletFamily>(m, "WaveletFamily")
        .value("HAAR", signal_processing::WaveletFamily::HAAR)
        .value("DAUBECHIES", signal_processing::WaveletFamily::DAUBECHIES)
        .value("SYMLET", signal_processing::WaveletFamily::SYMLET)
        .value("COIFLET", signal_processing::WaveletFamily::COIFLET)
        .value("BIORTHOGONAL", signal_processing::WaveletFamily::BIORTHOGONAL)
        .value("MEYER", signal_processing::WaveletFamily::MEYER)
        .value("MORLET", signal_processing::WaveletFamily::MORLET)
        .value("MEXICAN_HAT", signal_processing::WaveletFamily::MEXICAN_HAT);
    
    py::enum_<signal_processing::BoundaryMode>(m, "BoundaryMode")
        .value("ZERO_PADDING", signal_processing::BoundaryMode::ZERO_PADDING)
        .value("SYMMETRIC", signal_processing::BoundaryMode::SYMMETRIC)
        .value("PERIODIC", signal_processing::BoundaryMode::PERIODIC)
        .value("REFLECT", signal_processing::BoundaryMode::REFLECT);
    
    // Discrete Wavelet Transform
    py::class_<signal_processing::DiscreteWaveletTransform>(m, "DiscreteWaveletTransform")
        .def(py::init<signal_processing::WaveletFamily, int>(),
             py::arg("family") = signal_processing::WaveletFamily::DAUBECHIES,
             py::arg("vanishing_moments") = 4)
        .def("forward", [](signal_processing::DiscreteWaveletTransform& self, 
                         py::array_t<float> signal, int levels,
                         signal_processing::BoundaryMode mode) {
            auto vec_signal = numpy_to_vector(signal);
            auto result = self.forward(vec_signal, levels, mode);
            return transform_result_to_dict(result);
        }, py::arg("signal"), py::arg("levels") = 1, 
           py::arg("mode") = signal_processing::BoundaryMode::SYMMETRIC,
           "Perform forward DWT transform")
        .def("inverse", [](signal_processing::DiscreteWaveletTransform& self,
                         py::dict coeffs,
                         signal_processing::BoundaryMode mode) {
            auto result = dict_to_transform_result(coeffs);
            auto reconstructed = self.inverse(result, mode);
            return vector_to_numpy(reconstructed);
        }, py::arg("coeffs"), 
           py::arg("mode") = signal_processing::BoundaryMode::SYMMETRIC,
           "Perform inverse DWT transform")
        .def("get_decomposition_low_pass_filter", [](signal_processing::DiscreteWaveletTransform& self) {
            return vector_to_numpy(self.getDecompositionLowPassFilter());
        }, "Get decomposition low pass filter")
        .def("get_decomposition_high_pass_filter", [](signal_processing::DiscreteWaveletTransform& self) {
            return vector_to_numpy(self.getDecompositionHighPassFilter());
        }, "Get decomposition high pass filter")
        .def("get_reconstruction_low_pass_filter", [](signal_processing::DiscreteWaveletTransform& self) {
            return vector_to_numpy(self.getReconstructionLowPassFilter());
        }, "Get reconstruction low pass filter")
        .def("get_reconstruction_high_pass_filter", [](signal_processing::DiscreteWaveletTransform& self) {
            return vector_to_numpy(self.getReconstructionHighPassFilter());
        }, "Get reconstruction high pass filter");
    
    // Continuous Wavelet Transform
    py::class_<signal_processing::ContinuousWaveletTransform>(m, "ContinuousWaveletTransform")
        .def(py::init<signal_processing::WaveletFamily, int>(),
             py::arg("family") = signal_processing::WaveletFamily::MORLET,
             py::arg("vanishing_moments") = 1)
        .def("forward", [](signal_processing::ContinuousWaveletTransform& self, 
                         py::array_t<float> signal,
                         py::array_t<float> scales) {
            auto vec_signal = numpy_to_vector(signal);
            auto vec_scales = numpy_to_vector(scales);
            auto result = self.forward(vec_signal, vec_scales);
            return complex_vector_to_numpy(result);
        }, py::arg("signal"), py::arg("scales"),
           "Perform forward CWT transform")
        .def("inverse", [](signal_processing::ContinuousWaveletTransform& self,
                         py::array_t<std::complex<float>> coeffs,
                         py::array_t<float> scales) {
            auto vec_coeffs = numpy_to_complex_vector(coeffs);
            auto vec_scales = numpy_to_vector(scales);
            auto reconstructed = self.inverse(vec_coeffs, vec_scales);
            return vector_to_numpy(reconstructed);
        }, py::arg("coeffs"), py::arg("scales"),
           "Perform inverse CWT transform")
        .def("generate_scales", [](signal_processing::ContinuousWaveletTransform& self,
                                 int num_scales, float min_scale, float max_scale) {
            auto scales = self.generateScales(num_scales, min_scale, max_scale);
            return vector_to_numpy(scales);
        }, py::arg("num_scales"), py::arg("min_scale") = 1.0f, py::arg("max_scale") = 32.0f,
           "Generate logarithmically spaced scales for CWT");
    
    // Wavelet Packet Transform
    py::class_<signal_processing::WaveletPacketTransform>(m, "WaveletPacketTransform")
        .def(py::init<signal_processing::WaveletFamily, int>(),
             py::arg("family") = signal_processing::WaveletFamily::DAUBECHIES,
             py::arg("vanishing_moments") = 4)
        .def("forward", [](signal_processing::WaveletPacketTransform& self, 
                         py::array_t<float> signal, int levels,
                         signal_processing::BoundaryMode mode) {
            auto vec_signal = numpy_to_vector(signal);
            auto result = self.forward(vec_signal, levels, mode);
            return packet_result_to_dict(result);
        }, py::arg("signal"), py::arg("levels") = 1, 
           py::arg("mode") = signal_processing::BoundaryMode::SYMMETRIC,
           "Perform forward WPT transform")
        .def("inverse", [](signal_processing::WaveletPacketTransform& self,
                         py::dict result,
                         signal_processing::BoundaryMode mode) {
            auto packet_result = dict_to_packet_result(result);
            auto reconstructed = self.inverse(packet_result, mode);
            return vector_to_numpy(reconstructed);
        }, py::arg("result"), 
           py::arg("mode") = signal_processing::BoundaryMode::SYMMETRIC,
           "Perform inverse WPT transform")
        .def("get_decomposition_low_pass_filter", [](signal_processing::WaveletPacketTransform& self) {
            return vector_to_numpy(self.getDecompositionLowPassFilter());
        }, "Get decomposition low pass filter")
        .def("get_decomposition_high_pass_filter", [](signal_processing::WaveletPacketTransform& self) {
            return vector_to_numpy(self.getDecompositionHighPassFilter());
        }, "Get decomposition high pass filter")
        .def("get_reconstruction_low_pass_filter", [](signal_processing::WaveletPacketTransform& self) {
            return vector_to_numpy(self.getReconstructionLowPassFilter());
        }, "Get reconstruction low pass filter")
        .def("get_reconstruction_high_pass_filter", [](signal_processing::WaveletPacketTransform& self) {
            return vector_to_numpy(self.getReconstructionHighPassFilter());
        }, "Get reconstruction high pass filter");
    
    // Maximal Overlap DWT
    py::class_<signal_processing::MaximalOverlapDWT>(m, "MaximalOverlapDWT")
        .def(py::init<signal_processing::WaveletFamily, int>(),
             py::arg("family") = signal_processing::WaveletFamily::DAUBECHIES,
             py::arg("vanishing_moments") = 4)
        .def("forward", [](signal_processing::MaximalOverlapDWT& self, 
                         py::array_t<float> signal, int levels,
                         signal_processing::BoundaryMode mode) {
            auto vec_signal = numpy_to_vector(signal);
            auto result = self.forward(vec_signal, levels, mode);
            
            // Convert result to Python dict
            py::dict dict;
            
            // Convert wavelet coefficients
            py::list wavelet_list;
            for (const auto& coeff : result.detail_coefficients) {
                wavelet_list.append(vector_to_numpy(coeff));
            }
            dict["wavelet"] = wavelet_list;
            
            // Convert scaling coefficients
            py::list scaling_list;
            for (const auto& coeff : result.approximation_coefficients) {
                scaling_list.append(vector_to_numpy(coeff));
            }
            dict["scaling"] = scaling_list;
            
            return dict;
        }, py::arg("signal"), py::arg("levels") = 1, 
           py::arg("mode") = signal_processing::BoundaryMode::SYMMETRIC,
           "Perform forward MODWT transform")
        .def("inverse", [](signal_processing::MaximalOverlapDWT& self,
                         py::dict dict,
                         signal_processing::BoundaryMode mode) {
            signal_processing::WaveletTransformResult result;
            
            // Convert wavelet coefficients
            py::list wavelet_list = dict["wavelet"].cast<py::list>();
            result.detail_coefficients.resize(wavelet_list.size());
            for (size_t i = 0; i < wavelet_list.size(); i++) {
                result.detail_coefficients[i] = numpy_to_vector(wavelet_list[i].cast<py::array_t<float>>());
            }
            
            // Convert scaling coefficients
            py::list scaling_list = dict["scaling"].cast<py::list>();
            result.approximation_coefficients.resize(scaling_list.size());
            for (size_t i = 0; i < scaling_list.size(); i++) {
                result.approximation_coefficients[i] = numpy_to_vector(scaling_list[i].cast<py::array_t<float>>());
            }
            
            auto reconstructed = self.inverse(result, mode);
            return vector_to_numpy(reconstructed);
        }, py::arg("coeffs"), 
           py::arg("mode") = signal_processing::BoundaryMode::SYMMETRIC,
           "Perform inverse MODWT transform")
        .def("get_decomposition_low_pass_filter", [](signal_processing::MaximalOverlapDWT& self) {
            return vector_to_numpy(self.getDecompositionLowPassFilter());
        }, "Get decomposition low pass filter")
        .def("get_decomposition_high_pass_filter", [](signal_processing::MaximalOverlapDWT& self) {
            return vector_to_numpy(self.getDecompositionHighPassFilter());
        }, "Get decomposition high pass filter")
        .def("get_reconstruction_low_pass_filter", [](signal_processing::MaximalOverlapDWT& self) {
            return vector_to_numpy(self.getReconstructionLowPassFilter());
        }, "Get reconstruction low pass filter")
        .def("get_reconstruction_high_pass_filter", [](signal_processing::MaximalOverlapDWT& self) {
            return vector_to_numpy(self.getReconstructionHighPassFilter());
        }, "Get reconstruction high pass filter");
    
    // GPU-accelerated functions
#ifdef WITH_CUDA
    m.def("cuda_discrete_wavelet_transform", [](py::array_t<float> signal,
                                              py::array_t<float> decomp_low_pass,
                                              py::array_t<float> decomp_high_pass,
                                              int levels,
                                              signal_processing::BoundaryMode mode) {
        auto vec_signal = numpy_to_vector(signal);
        auto vec_decomp_low_pass = numpy_to_vector(decomp_low_pass);
        auto vec_decomp_high_pass = numpy_to_vector(decomp_high_pass);
        
        auto result = signal_processing::cuda_discrete_wavelet_transform(
            vec_signal, vec_decomp_low_pass, vec_decomp_high_pass, levels, mode);
        
        return transform_result_to_dict(result);
    }, py::arg("signal"), py::arg("decomp_low_pass"), py::arg("decomp_high_pass"),
       py::arg("levels"), py::arg("mode") = signal_processing::BoundaryMode::SYMMETRIC,
       "Perform GPU-accelerated discrete wavelet transform");
    
    m.def("cuda_inverse_discrete_wavelet_transform", [](py::dict dict,
                                                      py::array_t<float> recon_low_pass,
                                                      py::array_t<float> recon_high_pass,
                                                      signal_processing::BoundaryMode mode) {
        auto result = dict_to_transform_result(dict);
        auto vec_recon_low_pass = numpy_to_vector(recon_low_pass);
        auto vec_recon_high_pass = numpy_to_vector(recon_high_pass);
        
        auto reconstructed = signal_processing::cuda_inverse_discrete_wavelet_transform(
            result, vec_recon_low_pass, vec_recon_high_pass, mode);
        
        return vector_to_numpy(reconstructed);
    }, py::arg("coeffs"), py::arg("recon_low_pass"), py::arg("recon_high_pass"),
       py::arg("mode") = signal_processing::BoundaryMode::SYMMETRIC,
       "Perform GPU-accelerated inverse discrete wavelet transform");
    
    m.def("cuda_continuous_wavelet_transform", [](py::array_t<float> signal,
                                                py::array_t<float> scales,
                                                signal_processing::WaveletFamily family) {
        auto vec_signal = numpy_to_vector(signal);
        auto vec_scales = numpy_to_vector(scales);
        
        auto result = signal_processing::cuda_continuous_wavelet_transform(
            vec_signal, vec_scales, family);
        
        return complex_vector_to_numpy(result);
    }, py::arg("signal"), py::arg("scales"), 
       py::arg("family") = signal_processing::WaveletFamily::MORLET,
       "Perform GPU-accelerated continuous wavelet transform");
    
    m.def("cuda_wavelet_packet_transform", [](py::array_t<float> signal,
                                            py::array_t<float> decomp_low_pass,
                                            py::array_t<float> decomp_high_pass,
                                            int levels,
                                            signal_processing::BoundaryMode mode) {
        auto vec_signal = numpy_to_vector(signal);
        auto vec_decomp_low_pass = numpy_to_vector(decomp_low_pass);
        auto vec_decomp_high_pass = numpy_to_vector(decomp_high_pass);
        
        auto result = signal_processing::cuda_wavelet_packet_transform(
            vec_signal, vec_decomp_low_pass, vec_decomp_high_pass, levels, mode);
        
        return packet_result_to_dict(result);
    }, py::arg("signal"), py::arg("decomp_low_pass"), py::arg("decomp_high_pass"),
       py::arg("levels"), py::arg("mode") = signal_processing::BoundaryMode::SYMMETRIC,
       "Perform GPU-accelerated wavelet packet transform");
    
    m.def("cuda_inverse_wavelet_packet_transform", [](py::dict dict,
                                                    py::array_t<float> recon_low_pass,
                                                    py::array_t<float> recon_high_pass,
                                                    signal_processing::BoundaryMode mode) {
        auto result = dict_to_packet_result(dict);
        auto vec_recon_low_pass = numpy_to_vector(recon_low_pass);
        auto vec_recon_high_pass = numpy_to_vector(recon_high_pass);
        
        auto reconstructed = signal_processing::cuda_inverse_wavelet_packet_transform(
            result, vec_recon_low_pass, vec_recon_high_pass, mode);
        
        return vector_to_numpy(reconstructed);
    }, py::arg("coeffs"), py::arg("recon_low_pass"), py::arg("recon_high_pass"),
       py::arg("mode") = signal_processing::BoundaryMode::SYMMETRIC,
       "Perform GPU-accelerated inverse wavelet packet transform");
#endif
}