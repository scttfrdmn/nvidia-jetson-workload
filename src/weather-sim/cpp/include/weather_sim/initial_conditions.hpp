/**
 * @file initial_conditions.hpp
 * @brief Initial conditions for weather simulations.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#pragma once

#include <string>
#include <memory>
#include <random>
#include <functional>

#include "weather_sim.hpp"

namespace weather_sim {

/**
 * @brief Abstract factory for creating initial conditions.
 */
class InitialConditionFactory {
public:
    /**
     * @brief Get the singleton instance.
     * @return Reference to the singleton instance
     */
    static InitialConditionFactory& getInstance();
    
    /**
     * @brief Register an initial condition creator function.
     * @param name The name of the initial condition
     * @param creator Function to create the initial condition
     */
    void registerInitialCondition(
        const std::string& name,
        std::function<std::shared_ptr<InitialCondition>()> creator
    );
    
    /**
     * @brief Create an initial condition by name.
     * @param name The name of the initial condition to create
     * @return Shared pointer to the initial condition, or nullptr if not found
     */
    std::shared_ptr<InitialCondition> createInitialCondition(const std::string& name);
    
    /**
     * @brief Get a list of all available initial conditions.
     * @return Vector of initial condition names
     */
    std::vector<std::string> getAvailableInitialConditions() const;
    
private:
    // Private constructor for singleton
    InitialConditionFactory() = default;
    
    // No copy or move
    InitialConditionFactory(const InitialConditionFactory&) = delete;
    InitialConditionFactory& operator=(const InitialConditionFactory&) = delete;
    
    // Map of name to creator function
    std::map<std::string, std::function<std::shared_ptr<InitialCondition>()>> creators_;
};

/**
 * @brief Base class for parameterized initial conditions.
 */
class ParameterizedInitialCondition : public InitialCondition {
public:
    /**
     * @brief Set a parameter value.
     * @param name Parameter name
     * @param value Parameter value
     */
    template <typename T>
    void setParameter(const std::string& name, const T& value) {
        parameters_[name] = std::to_string(value);
    }
    
    /**
     * @brief Set a string parameter value.
     * @param name Parameter name
     * @param value Parameter value
     */
    void setStringParameter(const std::string& name, const std::string& value) {
        parameters_[name] = value;
    }
    
    /**
     * @brief Get a parameter value.
     * @param name Parameter name
     * @param default_value Default value to return if parameter not found
     * @return Parameter value, or default if not found
     */
    template <typename T>
    T getParameter(const std::string& name, const T& default_value) const {
        auto it = parameters_.find(name);
        if (it == parameters_.end()) {
            return default_value;
        }
        
        try {
            if constexpr (std::is_same_v<T, int>) {
                return std::stoi(it->second);
            } else if constexpr (std::is_same_v<T, float>) {
                return std::stof(it->second);
            } else if constexpr (std::is_same_v<T, double>) {
                return std::stod(it->second);
            } else if constexpr (std::is_same_v<T, bool>) {
                return it->second == "true" || it->second == "1";
            } else if constexpr (std::is_same_v<T, std::string>) {
                return it->second;
            } else {
                return default_value;
            }
        } catch (...) {
            return default_value;
        }
    }
    
    /**
     * @brief Get a string parameter value.
     * @param name Parameter name
     * @param default_value Default value to return if parameter not found
     * @return Parameter value, or default if not found
     */
    std::string getStringParameter(const std::string& name, const std::string& default_value) const {
        auto it = parameters_.find(name);
        if (it == parameters_.end()) {
            return default_value;
        }
        return it->second;
    }
    
    /**
     * @brief Check if a parameter exists.
     * @param name Parameter name
     * @return True if parameter exists
     */
    bool hasParameter(const std::string& name) const {
        return parameters_.find(name) != parameters_.end();
    }
    
    /**
     * @brief Get all parameter names.
     * @return Vector of parameter names
     */
    std::vector<std::string> getParameterNames() const {
        std::vector<std::string> names;
        for (const auto& param : parameters_) {
            names.push_back(param.first);
        }
        return names;
    }
    
protected:
    std::map<std::string, std::string> parameters_;
};

/**
 * @brief Initial condition with a uniform state.
 */
class UniformInitialCondition : public ParameterizedInitialCondition {
public:
    /**
     * @brief Construct a new Uniform Initial Condition.
     * @param u Initial u velocity
     * @param v Initial v velocity
     * @param h Initial height
     * @param p Initial pressure
     * @param t Initial temperature
     * @param q Initial humidity
     */
    UniformInitialCondition(
        scalar_t u = 0.0,
        scalar_t v = 0.0,
        scalar_t h = 10.0,
        scalar_t p = 1000.0,
        scalar_t t = 300.0,
        scalar_t q = 0.0
    );
    
    /**
     * @brief Initialize the weather grid with uniform conditions.
     * @param grid The grid to initialize
     */
    void initialize(WeatherGrid& grid) const override;
    
    /**
     * @brief Get the name of the initial condition.
     * @return Name of the initial condition
     */
    std::string getName() const override { return "uniform"; }
};

/**
 * @brief Initial condition with a random state.
 */
class RandomInitialCondition : public ParameterizedInitialCondition {
public:
    /**
     * @brief Construct a new Random Initial Condition.
     * @param seed Random seed
     * @param amplitude Amplitude of random perturbations
     */
    RandomInitialCondition(unsigned int seed = 0, scalar_t amplitude = 1.0);
    
    /**
     * @brief Initialize the weather grid with random conditions.
     * @param grid The grid to initialize
     */
    void initialize(WeatherGrid& grid) const override;
    
    /**
     * @brief Get the name of the initial condition.
     * @return Name of the initial condition
     */
    std::string getName() const override { return "random"; }
};

/**
 * @brief Initial condition with a zonal flow.
 */
class ZonalFlowInitialCondition : public ParameterizedInitialCondition {
public:
    /**
     * @brief Construct a new Zonal Flow Initial Condition.
     * @param u_max Maximum zonal (east-west) velocity
     * @param h_mean Mean height
     * @param beta Beta parameter (meridional gradient of Coriolis)
     */
    ZonalFlowInitialCondition(
        scalar_t u_max = 10.0,
        scalar_t h_mean = 10.0,
        scalar_t beta = 0.1
    );
    
    /**
     * @brief Initialize the weather grid with zonal flow.
     * @param grid The grid to initialize
     */
    void initialize(WeatherGrid& grid) const override;
    
    /**
     * @brief Get the name of the initial condition.
     * @return Name of the initial condition
     */
    std::string getName() const override { return "zonal_flow"; }
};

/**
 * @brief Initial condition with a single vortex.
 */
class VortexInitialCondition : public ParameterizedInitialCondition {
public:
    /**
     * @brief Construct a new Vortex Initial Condition.
     * @param x_center X coordinate of vortex center
     * @param y_center Y coordinate of vortex center
     * @param radius Radius of vortex
     * @param strength Strength of vortex
     * @param h_mean Mean height
     */
    VortexInitialCondition(
        scalar_t x_center = 0.5,
        scalar_t y_center = 0.5,
        scalar_t radius = 0.1,
        scalar_t strength = 10.0,
        scalar_t h_mean = 10.0
    );
    
    /**
     * @brief Initialize the weather grid with a vortex.
     * @param grid The grid to initialize
     */
    void initialize(WeatherGrid& grid) const override;
    
    /**
     * @brief Get the name of the initial condition.
     * @return Name of the initial condition
     */
    std::string getName() const override { return "vortex"; }
};

/**
 * @brief Initial condition with a jet stream.
 */
class JetStreamInitialCondition : public ParameterizedInitialCondition {
public:
    /**
     * @brief Construct a new Jet Stream Initial Condition.
     * @param y_center Y coordinate of jet center
     * @param width Width of jet
     * @param strength Strength of jet
     * @param h_mean Mean height
     */
    JetStreamInitialCondition(
        scalar_t y_center = 0.5,
        scalar_t width = 0.1,
        scalar_t strength = 10.0,
        scalar_t h_mean = 10.0
    );
    
    /**
     * @brief Initialize the weather grid with a jet stream.
     * @param grid The grid to initialize
     */
    void initialize(WeatherGrid& grid) const override;
    
    /**
     * @brief Get the name of the initial condition.
     * @return Name of the initial condition
     */
    std::string getName() const override { return "jet_stream"; }
};

/**
 * @brief Initial condition with a breaking wave.
 */
class BreakingWaveInitialCondition : public ParameterizedInitialCondition {
public:
    /**
     * @brief Construct a new Breaking Wave Initial Condition.
     * @param amplitude Wave amplitude
     * @param wavelength Wave length
     * @param h_mean Mean height
     */
    BreakingWaveInitialCondition(
        scalar_t amplitude = 1.0,
        scalar_t wavelength = 0.2,
        scalar_t h_mean = 10.0
    );
    
    /**
     * @brief Initialize the weather grid with a breaking wave.
     * @param grid The grid to initialize
     */
    void initialize(WeatherGrid& grid) const override;
    
    /**
     * @brief Get the name of the initial condition.
     * @return Name of the initial condition
     */
    std::string getName() const override { return "breaking_wave"; }
};

/**
 * @brief Initial condition with a front.
 */
class FrontInitialCondition : public ParameterizedInitialCondition {
public:
    /**
     * @brief Construct a new Front Initial Condition.
     * @param y_position Y position of front
     * @param width Width of front transition zone
     * @param temp_difference Temperature difference across front
     * @param wind_shear Wind shear across front
     */
    FrontInitialCondition(
        scalar_t y_position = 0.5,
        scalar_t width = 0.05,
        scalar_t temp_difference = 10.0,
        scalar_t wind_shear = 5.0
    );
    
    /**
     * @brief Initialize the weather grid with a front.
     * @param grid The grid to initialize
     */
    void initialize(WeatherGrid& grid) const override;
    
    /**
     * @brief Get the name of the initial condition.
     * @return Name of the initial condition
     */
    std::string getName() const override { return "front"; }
};

/**
 * @brief Initial condition with a mountain.
 */
class MountainInitialCondition : public ParameterizedInitialCondition {
public:
    /**
     * @brief Construct a new Mountain Initial Condition.
     * @param x_center X coordinate of mountain center
     * @param y_center Y coordinate of mountain center
     * @param radius Mountain radius
     * @param height Mountain height
     * @param u_base Base u velocity
     */
    MountainInitialCondition(
        scalar_t x_center = 0.3,
        scalar_t y_center = 0.5,
        scalar_t radius = 0.1,
        scalar_t height = 1.0,
        scalar_t u_base = 5.0
    );
    
    /**
     * @brief Initialize the weather grid with a mountain.
     * @param grid The grid to initialize
     */
    void initialize(WeatherGrid& grid) const override;
    
    /**
     * @brief Get the name of the initial condition.
     * @return Name of the initial condition
     */
    std::string getName() const override { return "mountain"; }
};

/**
 * @brief Initial condition with a realistic atmospheric profile.
 */
class AtmosphericProfileInitialCondition : public ParameterizedInitialCondition {
public:
    /**
     * @brief Construct a new Atmospheric Profile Initial Condition.
     * @param profile_name Name of the profile to use (e.g., "standard", "tropical", "polar")
     */
    explicit AtmosphericProfileInitialCondition(const std::string& profile_name = "standard");
    
    /**
     * @brief Initialize the weather grid with an atmospheric profile.
     * @param grid The grid to initialize
     */
    void initialize(WeatherGrid& grid) const override;
    
    /**
     * @brief Get the name of the initial condition.
     * @return Name of the initial condition
     */
    std::string getName() const override { return "atmospheric_profile"; }
    
private:
    void loadStandardProfile();
    void loadTropicalProfile();
    void loadPolarProfile();
    
    std::string profile_name_;
    std::vector<scalar_t> pressure_levels_;
    std::vector<scalar_t> temperature_profile_;
    std::vector<scalar_t> humidity_profile_;
    std::vector<scalar_t> u_wind_profile_;
    std::vector<scalar_t> v_wind_profile_;
};

// Register all available initial conditions
void registerAllInitialConditions();

} // namespace weather_sim