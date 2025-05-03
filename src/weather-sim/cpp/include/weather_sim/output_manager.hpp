/**
 * @file output_manager.hpp
 * @brief Output and visualization for weather simulations.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <fstream>
#include <functional>

#include "weather_sim.hpp"

namespace weather_sim {

/**
 * @brief Enumeration for output formats.
 */
enum class OutputFormat {
    CSV,      // CSV text format
    NetCDF,   // NetCDF binary format
    VTK,      // VTK format for visualization
    PNG,      // PNG image format
    Custom    // Custom output format
};

/**
 * @brief Configuration for output.
 */
struct OutputConfig {
    std::string output_dir = "./output";   // Directory for output files
    std::string prefix = "weather_sim";    // Prefix for output filenames
    OutputFormat format = OutputFormat::CSV; // Output format
    int output_interval = 10;              // Steps between outputs
    bool compress = false;                 // Whether to compress output
    bool include_diagnostics = true;       // Whether to include diagnostic fields
    std::vector<std::string> fields = {    // Fields to output
        "velocity", "height", "pressure", "temperature",
        "humidity", "vorticity", "divergence"
    };
};

/**
 * @brief Base class for CSV output.
 */
class CSVOutputManager : public OutputManager {
public:
    /**
     * @brief Construct a new CSV Output Manager.
     * @param config Output configuration
     */
    explicit CSVOutputManager(const OutputConfig& config = OutputConfig());
    
    /**
     * @brief Initialize the output manager.
     * @param simulation Reference to the simulation
     */
    void initialize(const WeatherSimulation& simulation) override;
    
    /**
     * @brief Write output for the current state.
     * @param simulation Reference to the simulation
     */
    void writeOutput(const WeatherSimulation& simulation) override;
    
    /**
     * @brief Finalize the output manager.
     * @param simulation Reference to the simulation
     */
    void finalize(const WeatherSimulation& simulation) override;
    
    /**
     * @brief Get the output configuration.
     * @return Output configuration
     */
    const OutputConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set the output configuration.
     * @param config New output configuration
     */
    void setConfig(const OutputConfig& config) { config_ = config; }
    
private:
    OutputConfig config_;
    std::string output_path_;
    int output_count_ = 0;
    
    void writeCSVHeader(std::ofstream& file, const WeatherGrid& grid);
    void writeCSVData(std::ofstream& file, const WeatherGrid& grid);
    void createOutputDirectory();
};

#ifdef HAVE_NETCDF
/**
 * @brief NetCDF output manager.
 */
class NetCDFOutputManager : public OutputManager {
public:
    /**
     * @brief Construct a new NetCDF Output Manager.
     * @param config Output configuration
     */
    explicit NetCDFOutputManager(const OutputConfig& config = OutputConfig());
    
    /**
     * @brief Initialize the output manager.
     * @param simulation Reference to the simulation
     */
    void initialize(const WeatherSimulation& simulation) override;
    
    /**
     * @brief Write output for the current state.
     * @param simulation Reference to the simulation
     */
    void writeOutput(const WeatherSimulation& simulation) override;
    
    /**
     * @brief Finalize the output manager.
     * @param simulation Reference to the simulation
     */
    void finalize(const WeatherSimulation& simulation) override;
    
    /**
     * @brief Get the output configuration.
     * @return Output configuration
     */
    const OutputConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set the output configuration.
     * @param config New output configuration
     */
    void setConfig(const OutputConfig& config) { config_ = config; }
    
private:
    OutputConfig config_;
    std::string output_path_;
    int output_count_ = 0;
    int ncid_ = -1;  // NetCDF file ID
    
    // NetCDF dimension IDs
    int dim_time_id_ = -1;
    int dim_x_id_ = -1;
    int dim_y_id_ = -1;
    int dim_level_id_ = -1;
    
    // NetCDF variable IDs
    std::map<std::string, int> var_ids_;
    
    void createNetCDFFile(const WeatherSimulation& simulation);
    void writeNetCDFData(const WeatherSimulation& simulation);
    void closeNetCDFFile();
};
#endif // HAVE_NETCDF

#ifdef HAVE_VTK
/**
 * @brief VTK output manager for visualization.
 */
class VTKOutputManager : public OutputManager {
public:
    /**
     * @brief Construct a new VTK Output Manager.
     * @param config Output configuration
     */
    explicit VTKOutputManager(const OutputConfig& config = OutputConfig());
    
    /**
     * @brief Initialize the output manager.
     * @param simulation Reference to the simulation
     */
    void initialize(const WeatherSimulation& simulation) override;
    
    /**
     * @brief Write output for the current state.
     * @param simulation Reference to the simulation
     */
    void writeOutput(const WeatherSimulation& simulation) override;
    
    /**
     * @brief Finalize the output manager.
     * @param simulation Reference to the simulation
     */
    void finalize(const WeatherSimulation& simulation) override;
    
    /**
     * @brief Get the output configuration.
     * @return Output configuration
     */
    const OutputConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set the output configuration.
     * @param config New output configuration
     */
    void setConfig(const OutputConfig& config) { config_ = config; }
    
private:
    OutputConfig config_;
    std::string output_path_;
    int output_count_ = 0;
    
    void writeVTKFile(const WeatherSimulation& simulation);
    void createOutputDirectory();
};
#endif // HAVE_VTK

/**
 * @brief Factory for creating output managers.
 */
class OutputManagerFactory {
public:
    /**
     * @brief Get the singleton instance.
     * @return Reference to the singleton instance
     */
    static OutputManagerFactory& getInstance();
    
    /**
     * @brief Register an output manager creator function.
     * @param format Output format
     * @param creator Function to create the output manager
     */
    void registerOutputManager(
        OutputFormat format,
        std::function<std::shared_ptr<OutputManager>(const OutputConfig&)> creator
    );
    
    /**
     * @brief Create an output manager for the specified format.
     * @param format Output format
     * @param config Output configuration
     * @return Shared pointer to the output manager, or nullptr if format not supported
     */
    std::shared_ptr<OutputManager> createOutputManager(
        OutputFormat format,
        const OutputConfig& config = OutputConfig()
    );
    
    /**
     * @brief Check if a format is supported.
     * @param format Output format to check
     * @return True if the format is supported
     */
    bool isFormatSupported(OutputFormat format) const;
    
    /**
     * @brief Get a list of all supported output formats.
     * @return Vector of supported output formats
     */
    std::vector<OutputFormat> getSupportedFormats() const;
    
private:
    // Private constructor for singleton
    OutputManagerFactory() = default;
    
    // No copy or move
    OutputManagerFactory(const OutputManagerFactory&) = delete;
    OutputManagerFactory& operator=(const OutputManagerFactory&) = delete;
    
    // Map of format to creator function
    std::map<OutputFormat, std::function<std::shared_ptr<OutputManager>(const OutputConfig&)>> creators_;
};

/**
 * @brief In-memory analysis and visualization manager.
 */
class VisualizationManager {
public:
    /**
     * @brief Construct a new Visualization Manager.
     */
    VisualizationManager();
    
    /**
     * @brief Initialize the visualization manager.
     * @param simulation Reference to the simulation
     */
    void initialize(const WeatherSimulation& simulation);
    
    /**
     * @brief Update the visualization with the current state.
     * @param simulation Reference to the simulation
     */
    void update(const WeatherSimulation& simulation);
    
    /**
     * @brief Generate a height field visualization.
     * @param simulation Reference to the simulation
     * @param filename Output filename
     * @param width Image width
     * @param height Image height
     * @param colormap Colormap name
     * @return True if successful
     */
    bool generateHeightFieldImage(
        const WeatherSimulation& simulation,
        const std::string& filename,
        int width = 800,
        int height = 600,
        const std::string& colormap = "viridis"
    );
    
    /**
     * @brief Generate a velocity field visualization.
     * @param simulation Reference to the simulation
     * @param filename Output filename
     * @param width Image width
     * @param height Image height
     * @param colormap Colormap name
     * @param streamlines Whether to use streamlines
     * @return True if successful
     */
    bool generateVelocityFieldImage(
        const WeatherSimulation& simulation,
        const std::string& filename,
        int width = 800,
        int height = 600,
        const std::string& colormap = "viridis",
        bool streamlines = true
    );
    
    /**
     * @brief Generate a vorticity visualization.
     * @param simulation Reference to the simulation
     * @param filename Output filename
     * @param width Image width
     * @param height Image height
     * @param colormap Colormap name
     * @return True if successful
     */
    bool generateVorticityImage(
        const WeatherSimulation& simulation,
        const std::string& filename,
        int width = 800,
        int height = 600,
        const std::string& colormap = "RdBu_r"
    );
    
    /**
     * @brief Generate a combined visualization with multiple fields.
     * @param simulation Reference to the simulation
     * @param filename Output filename
     * @param width Image width
     * @param height Image height
     * @return True if successful
     */
    bool generateCombinedImage(
        const WeatherSimulation& simulation,
        const std::string& filename,
        int width = 1200,
        int height = 900
    );
    
    /**
     * @brief Save performance metrics visualization.
     * @param simulation Reference to the simulation
     * @param filename Output filename
     * @param width Image width
     * @param height Image height
     * @return True if successful
     */
    bool savePerformanceMetricsImage(
        const WeatherSimulation& simulation,
        const std::string& filename,
        int width = 800,
        int height = 600
    );
    
private:
    struct FieldStatistics {
        scalar_t min_value = 0.0;
        scalar_t max_value = 0.0;
        scalar_t mean_value = 0.0;
        scalar_t std_dev = 0.0;
    };
    
    std::map<std::string, std::vector<scalar_t>> time_series_data_;
    std::map<std::string, FieldStatistics> field_statistics_;
    scalar_t start_time_ = 0.0;
    
    // Store performance metrics
    std::vector<double> step_times_;
    std::vector<double> memory_transfer_times_;
    std::vector<double> compute_times_;
    
    void updateFieldStatistics(const WeatherGrid& grid);
    void updateTimeSeries(const WeatherSimulation& simulation);
    void updatePerformanceMetrics(const WeatherSimulation& simulation);
    
    bool saveImage(
        const std::vector<unsigned char>& image_data,
        int width,
        int height,
        const std::string& filename
    );
    
    std::vector<unsigned char> generateHeightFieldImageData(
        const WeatherGrid& grid,
        int width,
        int height,
        const std::string& colormap
    );
    
    std::vector<unsigned char> generateVelocityFieldImageData(
        const WeatherGrid& grid,
        int width,
        int height,
        const std::string& colormap,
        bool streamlines
    );
    
    std::vector<unsigned char> generateVorticityImageData(
        const WeatherGrid& grid,
        int width,
        int height,
        const std::string& colormap
    );
};

// Helper function to register all output managers
void registerAllOutputManagers();

} // namespace weather_sim