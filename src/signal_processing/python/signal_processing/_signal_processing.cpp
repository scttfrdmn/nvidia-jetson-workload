#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "signal_processing/spectral_analysis.h"
#include "signal_processing/digital_filtering.h"
#include "signal_processing/time_frequency.h"

namespace py = pybind11;

// Helper function to convert NumPy arrays to STL vectors
template<typename T>
std::vector<T> numpy_to_vector(py::array_t<T> array) {
    auto buf = array.request();
    T* ptr = static_cast<T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + buf.size);
}

// Helper function to convert STL vectors to NumPy arrays
template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
    auto result = py::array_t<T>(vec.size());
    auto buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

// Helper function to convert complex STL vectors to NumPy arrays
py::array_t<std::complex<float>> complex_vector_to_numpy(const std::vector<std::complex<float>>& vec) {
    auto result = py::array_t<std::complex<float>>(vec.size());
    auto buf = result.request();
    std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

PYBIND11_MODULE(_signal_processing, m) {
    m.doc() = "Signal Processing CUDA-accelerated module";

    // ==================== Spectral Analysis ====================
    // Enum: WindowType
    py::enum_<WindowType>(m, "WindowType")
        .value("RECTANGULAR", WindowType::RECTANGULAR)
        .value("TRIANGULAR", WindowType::TRIANGULAR)
        .value("HANN", WindowType::HANN)
        .value("HAMMING", WindowType::HAMMING)
        .value("BLACKMAN", WindowType::BLACKMAN)
        .value("FLATTOP", WindowType::FLATTOP)
        .value("KAISER", WindowType::KAISER)
        .value("TUKEY", WindowType::TUKEY)
        .value("GAUSSIAN", WindowType::GAUSSIAN)
        .export_values();

    // Class: SpectralParams
    py::class_<SpectralParams>(m, "SpectralParams")
        .def(py::init<>())
        .def_readwrite("window_type", &SpectralParams::window_type)
        .def_readwrite("nfft", &SpectralParams::nfft)
        .def_readwrite("overlap", &SpectralParams::overlap)
        .def_readwrite("sample_rate", &SpectralParams::sample_rate)
        .def_readwrite("scaling", &SpectralParams::scaling)
        .def_readwrite("return_onesided", &SpectralParams::return_onesided);

    // Struct: PSDResult
    py::class_<PSDResult>(m, "PSDResult")
        .def(py::init<>())
        .def_readwrite("frequencies", &PSDResult::frequencies)
        .def_readwrite("psd", &PSDResult::psd);

    // Struct: CSDResult
    py::class_<CSDResult>(m, "CSDResult")
        .def(py::init<>())
        .def_readwrite("frequencies", &CSDResult::frequencies)
        .def_readwrite("coherence", &CSDResult::coherence);

    // Struct: SpectrogramResult
    py::class_<SpectrogramResult>(m, "SpectrogramResult")
        .def(py::init<>())
        .def_readwrite("times", &SpectrogramResult::times)
        .def_readwrite("frequencies", &SpectrogramResult::frequencies)
        .def_readwrite("spectrogram", &SpectrogramResult::spectrogram);

    // Class: FFT
    py::class_<FFT>(m, "FFT")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("forward_1d_real", [](FFT& self, py::array_t<float> input, bool normalize) {
            auto vec_input = numpy_to_vector<float>(input);
            auto result = self.forward_1d_real(vec_input, normalize);
            return complex_vector_to_numpy(result);
        }, py::arg("input"), py::arg("normalize") = false)
        .def("forward_1d_complex", [](FFT& self, py::array_t<std::complex<float>> input, bool normalize) {
            auto vec_input = numpy_to_vector<std::complex<float>>(input);
            auto result = self.forward_1d_complex(vec_input, normalize);
            return complex_vector_to_numpy(result);
        }, py::arg("input"), py::arg("normalize") = false)
        .def("inverse_1d_real", [](FFT& self, py::array_t<std::complex<float>> input, bool normalize) {
            auto vec_input = numpy_to_vector<std::complex<float>>(input);
            auto result = self.inverse_1d_real(vec_input, normalize);
            return vector_to_numpy(result);
        }, py::arg("input"), py::arg("normalize") = true)
        .def("inverse_1d_complex", [](FFT& self, py::array_t<std::complex<float>> input, bool normalize) {
            auto vec_input = numpy_to_vector<std::complex<float>>(input);
            auto result = self.inverse_1d_complex(vec_input, normalize);
            return complex_vector_to_numpy(result);
        }, py::arg("input"), py::arg("normalize") = true)
        .def("forward_2d_real", [](FFT& self, py::array_t<float> input, int rows, int cols, bool normalize) {
            auto vec_input = numpy_to_vector<float>(input);
            auto result = self.forward_2d_real(vec_input, rows, cols, normalize);
            return complex_vector_to_numpy(result);
        }, py::arg("input"), py::arg("rows"), py::arg("cols"), py::arg("normalize") = false)
        .def("forward_2d_complex", [](FFT& self, py::array_t<std::complex<float>> input, int rows, int cols, bool normalize) {
            auto vec_input = numpy_to_vector<std::complex<float>>(input);
            auto result = self.forward_2d_complex(vec_input, rows, cols, normalize);
            return complex_vector_to_numpy(result);
        }, py::arg("input"), py::arg("rows"), py::arg("cols"), py::arg("normalize") = false);

    // Class: SpectralAnalyzer
    py::class_<SpectralAnalyzer>(m, "SpectralAnalyzer")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("compute_psd", [](SpectralAnalyzer& self, py::array_t<float> signal, const SpectralParams& params) {
            auto vec_signal = numpy_to_vector<float>(signal);
            return self.compute_psd(vec_signal, params);
        }, py::arg("signal"), py::arg("params"))
        .def("compute_spectrogram", [](SpectralAnalyzer& self, py::array_t<float> signal, const SpectralParams& params) {
            auto vec_signal = numpy_to_vector<float>(signal);
            return self.compute_spectrogram(vec_signal, params);
        }, py::arg("signal"), py::arg("params"))
        .def("compute_coherence", [](SpectralAnalyzer& self, py::array_t<float> signal1, py::array_t<float> signal2, const SpectralParams& params) {
            auto vec_signal1 = numpy_to_vector<float>(signal1);
            auto vec_signal2 = numpy_to_vector<float>(signal2);
            return self.compute_coherence(vec_signal1, vec_signal2, params);
        }, py::arg("signal1"), py::arg("signal2"), py::arg("params"))
        .def("detect_peaks", [](SpectralAnalyzer& self, py::array_t<float> spectrum, py::array_t<float> frequencies, float threshold, int min_distance) {
            auto vec_spectrum = numpy_to_vector<float>(spectrum);
            auto vec_frequencies = numpy_to_vector<float>(frequencies);
            return self.detect_peaks(vec_spectrum, vec_frequencies, threshold, min_distance);
        }, py::arg("spectrum"), py::arg("frequencies"), py::arg("threshold") = 0.5f, py::arg("min_distance") = 1);

    // ==================== Digital Filtering ====================
    // Enum: FilterType
    py::enum_<FilterType>(m, "FilterType")
        .value("LOWPASS", FilterType::LOWPASS)
        .value("HIGHPASS", FilterType::HIGHPASS)
        .value("BANDPASS", FilterType::BANDPASS)
        .value("BANDSTOP", FilterType::BANDSTOP)
        .export_values();

    // Enum: FIRDesignMethod
    py::enum_<FIRDesignMethod>(m, "FIRDesignMethod")
        .value("WINDOW", FIRDesignMethod::WINDOW)
        .value("LEAST_SQUARES", FIRDesignMethod::LEAST_SQUARES)
        .value("PARKS_MCCLELLAN", FIRDesignMethod::PARKS_MCCLELLAN)
        .value("FREQUENCY_SAMPLING", FIRDesignMethod::FREQUENCY_SAMPLING)
        .export_values();

    // Enum: IIRDesignMethod
    py::enum_<IIRDesignMethod>(m, "IIRDesignMethod")
        .value("BUTTERWORTH", IIRDesignMethod::BUTTERWORTH)
        .value("CHEBYSHEV1", IIRDesignMethod::CHEBYSHEV1)
        .value("CHEBYSHEV2", IIRDesignMethod::CHEBYSHEV2)
        .value("ELLIPTIC", IIRDesignMethod::ELLIPTIC)
        .value("BESSEL", IIRDesignMethod::BESSEL)
        .export_values();

    // Enum: AdaptiveFilterType
    py::enum_<AdaptiveFilterType>(m, "AdaptiveFilterType")
        .value("LMS", AdaptiveFilterType::LMS)
        .value("NLMS", AdaptiveFilterType::NLMS)
        .value("RLS", AdaptiveFilterType::RLS)
        .value("KALMAN", AdaptiveFilterType::KALMAN)
        .export_values();

    // Class: FIRFilterParams
    py::class_<FIRFilterParams>(m, "FIRFilterParams")
        .def(py::init<>())
        .def_readwrite("filter_type", &FIRFilterParams::filter_type)
        .def_readwrite("cutoff_freqs", &FIRFilterParams::cutoff_freqs)
        .def_readwrite("window_type", &FIRFilterParams::window_type)
        .def_readwrite("num_taps", &FIRFilterParams::num_taps)
        .def_readwrite("design_method", &FIRFilterParams::design_method);

    // Class: IIRFilterParams
    py::class_<IIRFilterParams>(m, "IIRFilterParams")
        .def(py::init<>())
        .def_readwrite("filter_type", &IIRFilterParams::filter_type)
        .def_readwrite("cutoff_freqs", &IIRFilterParams::cutoff_freqs)
        .def_readwrite("order", &IIRFilterParams::order)
        .def_readwrite("design_method", &IIRFilterParams::design_method)
        .def_readwrite("ripple_db", &IIRFilterParams::ripple_db)
        .def_readwrite("stopband_atten_db", &IIRFilterParams::stopband_atten_db);

    // Class: AdaptiveFilterParams
    py::class_<AdaptiveFilterParams>(m, "AdaptiveFilterParams")
        .def(py::init<>())
        .def_readwrite("filter_length", &AdaptiveFilterParams::filter_length)
        .def_readwrite("filter_type", &AdaptiveFilterParams::filter_type)
        .def_readwrite("step_size", &AdaptiveFilterParams::step_size)
        .def_readwrite("forgetting_factor", &AdaptiveFilterParams::forgetting_factor)
        .def_readwrite("regularization", &AdaptiveFilterParams::regularization);

    // Class: MultirateFilterParams
    py::class_<MultirateFilterParams>(m, "MultirateFilterParams")
        .def(py::init<>())
        .def_readwrite("interpolation_factor", &MultirateFilterParams::interpolation_factor)
        .def_readwrite("decimation_factor", &MultirateFilterParams::decimation_factor)
        .def_readwrite("filter_params", &MultirateFilterParams::filter_params);

    // Class: FIRFilter
    py::class_<FIRFilter>(m, "FIRFilter")
        .def(py::init<const std::vector<float>&, int>(), py::arg("coefficients"), py::arg("device_id") = 0)
        .def(py::init<const FIRFilterParams&, float, int>(), py::arg("params"), py::arg("sample_rate"), py::arg("device_id") = 0)
        .def("filter", [](FIRFilter& self, py::array_t<float> signal) {
            auto vec_signal = numpy_to_vector<float>(signal);
            auto result = self.filter(vec_signal);
            return vector_to_numpy(result);
        }, py::arg("signal"))
        .def("reset", &FIRFilter::reset)
        .def("get_coefficients", &FIRFilter::get_coefficients)
        .def("get_frequency_response", &FIRFilter::get_frequency_response, py::arg("num_points") = 512)
        .def("get_phase_response", &FIRFilter::get_phase_response, py::arg("num_points") = 512)
        .def("get_step_response", &FIRFilter::get_step_response, py::arg("num_points") = 100)
        .def("get_impulse_response", &FIRFilter::get_impulse_response, py::arg("num_points") = 100);

    // Class: IIRFilter
    py::class_<IIRFilter>(m, "IIRFilter")
        .def(py::init<const std::vector<float>&, const std::vector<float>&, int>(), py::arg("a"), py::arg("b"), py::arg("device_id") = 0)
        .def(py::init<const IIRFilterParams&, float, int>(), py::arg("params"), py::arg("sample_rate"), py::arg("device_id") = 0)
        .def("filter", [](IIRFilter& self, py::array_t<float> signal) {
            auto vec_signal = numpy_to_vector<float>(signal);
            auto result = self.filter(vec_signal);
            return vector_to_numpy(result);
        }, py::arg("signal"))
        .def("filter_sos", [](IIRFilter& self, py::array_t<float> signal) {
            auto vec_signal = numpy_to_vector<float>(signal);
            auto result = self.filter_sos(vec_signal);
            return vector_to_numpy(result);
        }, py::arg("signal"))
        .def("reset", &IIRFilter::reset)
        .def("get_coefficients", &IIRFilter::get_coefficients)
        .def("get_frequency_response", &IIRFilter::get_frequency_response, py::arg("num_points") = 512)
        .def("get_phase_response", &IIRFilter::get_phase_response, py::arg("num_points") = 512)
        .def("is_stable", &IIRFilter::is_stable);

    // Class: AdaptiveFilter
    py::class_<AdaptiveFilter>(m, "AdaptiveFilter")
        .def(py::init<const AdaptiveFilterParams&, int>(), py::arg("params"), py::arg("device_id") = 0)
        .def("filter", [](AdaptiveFilter& self, py::array_t<float> input_signal, py::array_t<float> desired_signal) {
            auto vec_input = numpy_to_vector<float>(input_signal);
            auto vec_desired = numpy_to_vector<float>(desired_signal);
            auto result = self.filter(vec_input, vec_desired);
            return py::make_tuple(vector_to_numpy(result.first), vector_to_numpy(result.second));
        }, py::arg("input_signal"), py::arg("desired_signal"))
        .def("get_coefficients", &AdaptiveFilter::get_coefficients)
        .def("get_learning_curve", &AdaptiveFilter::get_learning_curve)
        .def("reset", &AdaptiveFilter::reset);

    // Class: MultirateFilter
    py::class_<MultirateFilter>(m, "MultirateFilter")
        .def(py::init<const MultirateFilterParams&, int>(), py::arg("params"), py::arg("device_id") = 0)
        .def("upsample", [](MultirateFilter& self, py::array_t<float> signal) {
            auto vec_signal = numpy_to_vector<float>(signal);
            auto result = self.upsample(vec_signal);
            return vector_to_numpy(result);
        }, py::arg("signal"))
        .def("downsample", [](MultirateFilter& self, py::array_t<float> signal) {
            auto vec_signal = numpy_to_vector<float>(signal);
            auto result = self.downsample(vec_signal);
            return vector_to_numpy(result);
        }, py::arg("signal"))
        .def("resample", [](MultirateFilter& self, py::array_t<float> signal) {
            auto vec_signal = numpy_to_vector<float>(signal);
            auto result = self.resample(vec_signal);
            return vector_to_numpy(result);
        }, py::arg("signal"))
        .def("reset", &MultirateFilter::reset)
        .def("get_coefficients", &MultirateFilter::get_coefficients);

    // Add filters submodule
    py::module filters = m.def_submodule("filters", "Digital filtering functions");
    
    filters.def("median_filter", [](py::array_t<float> signal, int kernel_size, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = median_filter(vec_signal, kernel_size, device_id);
        return vector_to_numpy(result);
    }, py::arg("signal"), py::arg("kernel_size"), py::arg("device_id") = 0);
    
    filters.def("convolve", [](py::array_t<float> signal, py::array_t<float> kernel, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto vec_kernel = numpy_to_vector<float>(kernel);
        auto result = convolve(vec_signal, vec_kernel, device_id);
        return vector_to_numpy(result);
    }, py::arg("signal"), py::arg("kernel"), py::arg("device_id") = 0);
    
    filters.def("savitzky_golay", [](py::array_t<float> signal, int window_size, int poly_order, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = savitzky_golay(vec_signal, window_size, poly_order, device_id);
        return vector_to_numpy(result);
    }, py::arg("signal"), py::arg("window_size"), py::arg("poly_order"), py::arg("device_id") = 0);
    
    filters.def("wiener_filter", [](py::array_t<float> signal, float noise_power, int kernel_size, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = wiener_filter(vec_signal, noise_power, kernel_size, device_id);
        return vector_to_numpy(result);
    }, py::arg("signal"), py::arg("noise_power"), py::arg("kernel_size") = 5, py::arg("device_id") = 0);
    
    filters.def("kalman_filter", [](py::array_t<float> signal, float process_variance, float measurement_variance, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = kalman_filter(vec_signal, process_variance, measurement_variance, device_id);
        return vector_to_numpy(result);
    }, py::arg("signal"), py::arg("process_variance"), py::arg("measurement_variance"), py::arg("device_id") = 0);
    
    filters.def("bilateral_filter", [](py::array_t<float> signal, float sigma_d, float sigma_r, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = bilateral_filter(vec_signal, sigma_d, sigma_r, device_id);
        return vector_to_numpy(result);
    }, py::arg("signal"), py::arg("sigma_d"), py::arg("sigma_r"), py::arg("device_id") = 0);

    // ==================== Time-Frequency Analysis ====================
    // Enum: WaveletType
    py::enum_<WaveletType>(m, "WaveletType")
        .value("MORLET", WaveletType::MORLET)
        .value("MEXICAN_HAT", WaveletType::MEXICAN_HAT)
        .value("PAUL", WaveletType::PAUL)
        .value("DOG", WaveletType::DOG)
        .value("HAAR", WaveletType::HAAR)
        .value("DB4", WaveletType::DB4)
        .export_values();

    // Class: STFTParams
    py::class_<STFTParams>(m, "STFTParams")
        .def(py::init<>())
        .def_readwrite("window_size", &STFTParams::window_size)
        .def_readwrite("hop_size", &STFTParams::hop_size)
        .def_readwrite("window_type", &STFTParams::window_type)
        .def_readwrite("fft_size", &STFTParams::fft_size)
        .def_readwrite("center", &STFTParams::center)
        .def_readwrite("pad_mode", &STFTParams::pad_mode);

    // Class: CWTParams
    py::class_<CWTParams>(m, "CWTParams")
        .def(py::init<>())
        .def_readwrite("wavelet_type", &CWTParams::wavelet_type)
        .def_readwrite("wavelet_param", &CWTParams::wavelet_param)
        .def_readwrite("num_scales", &CWTParams::num_scales)
        .def_readwrite("min_scale", &CWTParams::min_scale)
        .def_readwrite("max_scale", &CWTParams::max_scale)
        .def_readwrite("normalize_scales", &CWTParams::normalize_scales);

    // Class: DWTParams
    py::class_<DWTParams>(m, "DWTParams")
        .def(py::init<>())
        .def_readwrite("wavelet_type", &DWTParams::wavelet_type)
        .def_readwrite("levels", &DWTParams::levels)
        .def_readwrite("mode", &DWTParams::mode)
        .def_readwrite("use_swt", &DWTParams::use_swt);

    // Struct: STFTResult
    py::class_<STFTResult>(m, "STFTResult")
        .def(py::init<>())
        .def_readwrite("spectrogram", &STFTResult::spectrogram)
        .def_readwrite("times", &STFTResult::times)
        .def_readwrite("frequencies", &STFTResult::frequencies)
        .def_readwrite("sample_rate", &STFTResult::sample_rate);

    // Struct: CWTResult
    py::class_<CWTResult>(m, "CWTResult")
        .def(py::init<>())
        .def_readwrite("scalogram", &CWTResult::scalogram)
        .def_readwrite("times", &CWTResult::times)
        .def_readwrite("scales", &CWTResult::scales)
        .def_readwrite("frequencies", &CWTResult::frequencies);

    // Struct: DWTResult
    py::class_<DWTResult>(m, "DWTResult")
        .def(py::init<>())
        .def_readwrite("coeffs", &DWTResult::coeffs)
        .def_readwrite("lengths", &DWTResult::lengths)
        .def_readwrite("levels", &DWTResult::levels);

    // Struct: IMF
    py::class_<IMF>(m, "IMF")
        .def(py::init<>())
        .def(py::init<std::vector<float>, float, std::vector<float>>())
        .def_readwrite("signal", &IMF::signal)
        .def_readwrite("mean_period", &IMF::mean_period)
        .def_readwrite("instantaneous_frequency", &IMF::instantaneous_frequency);

    // Struct: EMDResult
    py::class_<EMDResult>(m, "EMDResult")
        .def(py::init<>())
        .def_readwrite("imfs", &EMDResult::imfs)
        .def_readwrite("residue", &EMDResult::residue)
        .def_readwrite("num_imfs", &EMDResult::num_imfs);

    // Class: STFT
    py::class_<STFT>(m, "STFT")
        .def(py::init<const STFTParams&, int>(), py::arg("params"), py::arg("device_id") = 0)
        .def("transform", [](STFT& self, py::array_t<float> signal, float sample_rate) {
            auto vec_signal = numpy_to_vector<float>(signal);
            return self.transform(vec_signal, sample_rate);
        }, py::arg("signal"), py::arg("sample_rate"))
        .def("inverse_transform", &STFT::inverse_transform, py::arg("stft_result"));

    // Class: CWT
    py::class_<CWT>(m, "CWT")
        .def(py::init<const CWTParams&, int>(), py::arg("params"), py::arg("device_id") = 0)
        .def("transform", [](CWT& self, py::array_t<float> signal, float sample_rate) {
            auto vec_signal = numpy_to_vector<float>(signal);
            return self.transform(vec_signal, sample_rate);
        }, py::arg("signal"), py::arg("sample_rate"));

    // Class: DWT
    py::class_<DWT>(m, "DWT")
        .def(py::init<const DWTParams&, int>(), py::arg("params"), py::arg("device_id") = 0)
        .def("transform", [](DWT& self, py::array_t<float> signal) {
            auto vec_signal = numpy_to_vector<float>(signal);
            return self.transform(vec_signal);
        }, py::arg("signal"))
        .def("inverse_transform", &DWT::inverse_transform, py::arg("dwt_result"))
        .def("denoise", [](DWT& self, py::array_t<float> signal, float threshold, const std::string& threshold_mode) {
            auto vec_signal = numpy_to_vector<float>(signal);
            return vector_to_numpy(self.denoise(vec_signal, threshold, threshold_mode));
        }, py::arg("signal"), py::arg("threshold"), py::arg("threshold_mode") = "soft");

    // Class: WignerVille
    py::class_<WignerVille>(m, "WignerVille")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("transform", [](WignerVille& self, py::array_t<float> signal, float sample_rate) {
            auto vec_signal = numpy_to_vector<float>(signal);
            return self.transform(vec_signal, sample_rate);
        }, py::arg("signal"), py::arg("sample_rate"))
        .def("transform_pseudo", [](WignerVille& self, py::array_t<float> signal, float sample_rate, int window_size) {
            auto vec_signal = numpy_to_vector<float>(signal);
            return self.transform_pseudo(vec_signal, sample_rate, window_size);
        }, py::arg("signal"), py::arg("sample_rate"), py::arg("window_size") = 127);

    // Class: EMD
    py::class_<EMD>(m, "EMD")
        .def(py::init<int, int>(), py::arg("max_imfs") = 0, py::arg("device_id") = 0)
        .def("decompose", [](EMD& self, py::array_t<float> signal, float sample_rate) {
            auto vec_signal = numpy_to_vector<float>(signal);
            return self.decompose(vec_signal, sample_rate);
        }, py::arg("signal"), py::arg("sample_rate"))
        .def("hilbert_huang_spectrum", &EMD::hilbert_huang_spectrum, 
            py::arg("emd_result"), py::arg("sample_rate"), py::arg("num_freqs") = 256)
        .def("reconstruct", &EMD::reconstruct, 
            py::arg("emd_result"), py::arg("imf_indices") = std::vector<int>());

    // Add time_frequency submodule
    py::module time_frequency = m.def_submodule("time_frequency", "Time-frequency analysis functions");
    
    time_frequency.def("spectrogram", [](py::array_t<float> signal, float sample_rate, int window_size, int hop_size, 
                                         WindowType window_type, bool log_scale, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = spectrogram(vec_signal, sample_rate, window_size, hop_size, window_type, log_scale, device_id);
        return result;
    }, py::arg("signal"), py::arg("sample_rate"), py::arg("window_size") = 1024, py::arg("hop_size") = 256, 
       py::arg("window_type") = WindowType::HANN, py::arg("log_scale") = true, py::arg("device_id") = 0);
    
    time_frequency.def("scalogram", [](py::array_t<float> signal, float sample_rate, WaveletType wavelet_type, 
                                      int num_scales, bool log_scale, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = scalogram(vec_signal, sample_rate, wavelet_type, num_scales, log_scale, device_id);
        return result;
    }, py::arg("signal"), py::arg("sample_rate"), py::arg("wavelet_type") = WaveletType::MORLET, 
       py::arg("num_scales") = 32, py::arg("log_scale") = true, py::arg("device_id") = 0);
    
    time_frequency.def("mel_spectrogram", [](py::array_t<float> signal, float sample_rate, int n_fft, int hop_size, 
                                           int n_mels, float fmin, float fmax, bool log_scale, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = mel_spectrogram(vec_signal, sample_rate, n_fft, hop_size, n_mels, fmin, fmax, log_scale, device_id);
        return result;
    }, py::arg("signal"), py::arg("sample_rate"), py::arg("n_fft") = 2048, py::arg("hop_size") = 512, 
       py::arg("n_mels") = 128, py::arg("fmin") = 0.0f, py::arg("fmax") = 0.0f, 
       py::arg("log_scale") = true, py::arg("device_id") = 0);
    
    time_frequency.def("mfcc", [](py::array_t<float> signal, float sample_rate, int n_mfcc, int n_fft, 
                                int hop_size, int n_mels, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = mfcc(vec_signal, sample_rate, n_mfcc, n_fft, hop_size, n_mels, device_id);
        return result;
    }, py::arg("signal"), py::arg("sample_rate"), py::arg("n_mfcc") = 13, py::arg("n_fft") = 2048, 
       py::arg("hop_size") = 512, py::arg("n_mels") = 128, py::arg("device_id") = 0);
    
    time_frequency.def("chroma", [](py::array_t<float> signal, float sample_rate, int n_fft, int hop_size, 
                                  int n_chroma, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = chroma(vec_signal, sample_rate, n_fft, hop_size, n_chroma, device_id);
        return result;
    }, py::arg("signal"), py::arg("sample_rate"), py::arg("n_fft") = 2048, py::arg("hop_size") = 512, 
       py::arg("n_chroma") = 12, py::arg("device_id") = 0);
    
    time_frequency.def("hilbert_transform", [](py::array_t<float> signal, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = hilbert_transform(vec_signal, device_id);
        return complex_vector_to_numpy(result);
    }, py::arg("signal"), py::arg("device_id") = 0);
    
    time_frequency.def("instantaneous_frequency", [](py::array_t<std::complex<float>> analytic_signal, 
                                                  float sample_rate, int device_id) {
        auto vec_signal = numpy_to_vector<std::complex<float>>(analytic_signal);
        auto result = instantaneous_frequency(vec_signal, sample_rate, device_id);
        return vector_to_numpy(result);
    }, py::arg("analytic_signal"), py::arg("sample_rate"), py::arg("device_id") = 0);
    
    time_frequency.def("reassigned_spectrogram", [](py::array_t<float> signal, float sample_rate, int window_size, 
                                                 int hop_size, WindowType window_type, int device_id) {
        auto vec_signal = numpy_to_vector<float>(signal);
        auto result = reassigned_spectrogram(vec_signal, sample_rate, window_size, hop_size, window_type, device_id);
        return result;
    }, py::arg("signal"), py::arg("sample_rate"), py::arg("window_size") = 1024, py::arg("hop_size") = 256, 
       py::arg("window_type") = WindowType::HANN, py::arg("device_id") = 0);
}