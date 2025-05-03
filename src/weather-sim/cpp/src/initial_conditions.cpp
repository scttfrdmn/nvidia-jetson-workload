/**
 * @file initial_conditions.cpp
 * @brief Implementation of initial conditions for weather simulations.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include "../include/weather_sim/initial_conditions.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace weather_sim {

// InitialConditionFactory implementation
InitialConditionFactory& InitialConditionFactory::getInstance() {
    static InitialConditionFactory instance;
    return instance;
}

void InitialConditionFactory::registerInitialCondition(
    const std::string& name,
    std::function<std::shared_ptr<InitialCondition>()> creator
) {
    creators_[name] = creator;
}

std::shared_ptr<InitialCondition> InitialConditionFactory::createInitialCondition(
    const std::string& name
) {
    auto it = creators_.find(name);
    if (it == creators_.end()) {
        return nullptr;
    }
    
    return it->second();
}

std::vector<std::string> InitialConditionFactory::getAvailableInitialConditions() const {
    std::vector<std::string> names;
    for (const auto& entry : creators_) {
        names.push_back(entry.first);
    }
    return names;
}

// UniformInitialCondition implementation
UniformInitialCondition::UniformInitialCondition(
    scalar_t u, scalar_t v, scalar_t h, scalar_t p, scalar_t t, scalar_t q
) {
    setParameter("u", u);
    setParameter("v", v);
    setParameter("h", h);
    setParameter("p", p);
    setParameter("t", t);
    setParameter("q", q);
}

void UniformInitialCondition::initialize(WeatherGrid& grid) const {
    scalar_t u = getParameter("u", 0.0f);
    scalar_t v = getParameter("v", 0.0f);
    scalar_t h = getParameter("h", 10.0f);
    scalar_t p = getParameter("p", 1000.0f);
    scalar_t t = getParameter("t", 300.0f);
    scalar_t q = getParameter("q", 0.0f);
    
    auto& velocity = grid.getVelocityField();
    auto& height = grid.getHeightField();
    auto& pressure = grid.getPressureField();
    auto& temperature = grid.getTemperatureField();
    auto& humidity = grid.getHumidityField();
    
    index_t width = grid.getWidth();
    index_t height_dim = grid.getHeight();
    
    // Set uniform values across the grid
    for (index_t y = 0; y < height_dim; ++y) {
        for (index_t x = 0; x < width; ++x) {
            velocity.set(x, y, u, v);
            height(x, y) = h;
            pressure(x, y) = p;
            temperature(x, y) = t;
            humidity(x, y) = q;
        }
    }
    
    // Calculate diagnostics
    grid.calculateDiagnostics();
}

// RandomInitialCondition implementation
RandomInitialCondition::RandomInitialCondition(unsigned int seed, scalar_t amplitude) {
    setParameter("seed", static_cast<int>(seed));
    setParameter("amplitude", amplitude);
}

void RandomInitialCondition::initialize(WeatherGrid& grid) const {
    int seed = getParameter("seed", 0);
    scalar_t amplitude = getParameter("amplitude", 1.0f);
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<scalar_t> dist(-amplitude, amplitude);
    
    auto& velocity = grid.getVelocityField();
    auto& height = grid.getHeightField();
    
    index_t width = grid.getWidth();
    index_t height_dim = grid.getHeight();
    
    // Set random perturbations
    for (index_t y = 0; y < height_dim; ++y) {
        for (index_t x = 0; x < width; ++x) {
            scalar_t u = dist(rng);
            scalar_t v = dist(rng);
            scalar_t h = 10.0f + dist(rng);
            
            velocity.set(x, y, u, v);
            height(x, y) = h;
        }
    }
    
    // Calculate diagnostics
    grid.calculateDiagnostics();
}

// ZonalFlowInitialCondition implementation
ZonalFlowInitialCondition::ZonalFlowInitialCondition(
    scalar_t u_max, scalar_t h_mean, scalar_t beta
) {
    setParameter("u_max", u_max);
    setParameter("h_mean", h_mean);
    setParameter("beta", beta);
}

void ZonalFlowInitialCondition::initialize(WeatherGrid& grid) const {
    scalar_t u_max = getParameter("u_max", 10.0f);
    scalar_t h_mean = getParameter("h_mean", 10.0f);
    scalar_t beta = getParameter("beta", 0.1f);
    
    auto& velocity = grid.getVelocityField();
    auto& height = grid.getHeightField();
    
    index_t width = grid.getWidth();
    index_t height_dim = grid.getHeight();
    
    // Create a zonal (east-west) flow with meridional (north-south) variation
    for (index_t y = 0; y < height_dim; ++y) {
        // Normalize y coordinate to [0, 1]
        scalar_t y_norm = static_cast<scalar_t>(y) / (height_dim - 1);
        
        // Calculate zonal velocity: maximum at mid-latitude, zero at poles
        scalar_t u = u_max * std::sin(M_PI * y_norm);
        
        // Calculate corresponding height field for geostrophic balance
        scalar_t h_base = h_mean;
        
        // Apply to each column
        for (index_t x = 0; x < width; ++x) {
            velocity.set(x, y, u, 0.0f);
            
            // Height field varies with latitude to balance the zonal flow
            // h_y = -f*u/g (geostrophic balance)
            // Integrate h_y to get h(y)
            // f = f0 + beta*y
            scalar_t f = 1.0e-4f + beta * (y_norm - 0.5f);
            scalar_t h = h_base - 0.5f * f * u * u / 9.81f;
            
            height(x, y) = h;
        }
    }
    
    // Calculate diagnostics
    grid.calculateDiagnostics();
}

// VortexInitialCondition implementation
VortexInitialCondition::VortexInitialCondition(
    scalar_t x_center, scalar_t y_center, scalar_t radius, scalar_t strength, scalar_t h_mean
) {
    setParameter("x_center", x_center);
    setParameter("y_center", y_center);
    setParameter("radius", radius);
    setParameter("strength", strength);
    setParameter("h_mean", h_mean);
}

void VortexInitialCondition::initialize(WeatherGrid& grid) const {
    scalar_t x_center = getParameter("x_center", 0.5f);
    scalar_t y_center = getParameter("y_center", 0.5f);
    scalar_t radius = getParameter("radius", 0.1f);
    scalar_t strength = getParameter("strength", 10.0f);
    scalar_t h_mean = getParameter("h_mean", 10.0f);
    
    auto& velocity = grid.getVelocityField();
    auto& height = grid.getHeightField();
    
    index_t width = grid.getWidth();
    index_t height_dim = grid.getHeight();
    
    scalar_t x_center_grid = x_center * (width - 1);
    scalar_t y_center_grid = y_center * (height_dim - 1);
    scalar_t radius_grid = radius * std::min(width, height_dim);
    
    // Create a circular vortex
    for (index_t y = 0; y < height_dim; ++y) {
        for (index_t x = 0; x < width; ++x) {
            // Calculate distance from vortex center
            scalar_t dx = x - x_center_grid;
            scalar_t dy = y - y_center_grid;
            scalar_t r = std::sqrt(dx * dx + dy * dy);
            
            // Velocity field for vortex
            scalar_t angular_velocity = 0.0f;
            scalar_t h = h_mean;
            
            if (r > 0.0f && r <= radius_grid) {
                // Smooth vortex profile (Rankine vortex)
                scalar_t r_norm = r / radius_grid;
                angular_velocity = strength * r_norm * std::exp(1.0f - r_norm * r_norm);
                
                // Adjust height field for cyclostrophic balance
                // dh/dr = v^2 / (g*r)
                h = h_mean - 0.5f * angular_velocity * angular_velocity / 9.81f;
            }
            
            // Convert angular velocity to Cartesian components
            scalar_t u = -angular_velocity * dy / std::max(r, 1.0e-6f);
            scalar_t v = angular_velocity * dx / std::max(r, 1.0e-6f);
            
            velocity.set(x, y, u, v);
            height(x, y) = h;
        }
    }
    
    // Calculate diagnostics
    grid.calculateDiagnostics();
}

// JetStreamInitialCondition implementation
JetStreamInitialCondition::JetStreamInitialCondition(
    scalar_t y_center, scalar_t width, scalar_t strength, scalar_t h_mean
) {
    setParameter("y_center", y_center);
    setParameter("width", width);
    setParameter("strength", strength);
    setParameter("h_mean", h_mean);
}

void JetStreamInitialCondition::initialize(WeatherGrid& grid) const {
    scalar_t y_center = getParameter("y_center", 0.5f);
    scalar_t width_param = getParameter("width", 0.1f);
    scalar_t strength = getParameter("strength", 10.0f);
    scalar_t h_mean = getParameter("h_mean", 10.0f);
    
    auto& velocity = grid.getVelocityField();
    auto& height = grid.getHeightField();
    
    index_t grid_width = grid.getWidth();
    index_t grid_height = grid.getHeight();
    
    scalar_t y_center_grid = y_center * (grid_height - 1);
    scalar_t width_grid = width_param * grid_height;
    
    // Create a jet stream (strong zonal flow confined to a narrow band)
    for (index_t y = 0; y < grid_height; ++y) {
        // Distance from jet center
        scalar_t dy = y - y_center_grid;
        
        // Jet profile (Gaussian)
        scalar_t u = strength * std::exp(-(dy * dy) / (2.0f * width_grid * width_grid));
        
        // Adjust height field for geostrophic balance
        // dh/dy = -f*u/g
        scalar_t dh_dy = -1.0e-4f * u / 9.81f;
        
        for (index_t x = 0; x < grid_width; ++x) {
            velocity.set(x, y, u, 0.0f);
            
            // Height field varies to balance the jet (simplified)
            scalar_t h = h_mean + dh_dy * dy;
            height(x, y) = h;
        }
    }
    
    // Calculate diagnostics
    grid.calculateDiagnostics();
}

// BreakingWaveInitialCondition implementation
BreakingWaveInitialCondition::BreakingWaveInitialCondition(
    scalar_t amplitude, scalar_t wavelength, scalar_t h_mean
) {
    setParameter("amplitude", amplitude);
    setParameter("wavelength", wavelength);
    setParameter("h_mean", h_mean);
}

void BreakingWaveInitialCondition::initialize(WeatherGrid& grid) const {
    scalar_t amplitude = getParameter("amplitude", 1.0f);
    scalar_t wavelength = getParameter("wavelength", 0.2f);
    scalar_t h_mean = getParameter("h_mean", 10.0f);
    
    auto& velocity = grid.getVelocityField();
    auto& height = grid.getHeightField();
    
    index_t grid_width = grid.getWidth();
    index_t grid_height = grid.getHeight();
    
    // Wavelength in grid units
    scalar_t wave_k = 2.0f * M_PI / (wavelength * grid_width);
    
    // Create a breaking wave pattern
    for (index_t y = 0; y < grid_height; ++y) {
        // Normalized y-coordinate
        scalar_t y_norm = static_cast<scalar_t>(y) / (grid_height - 1);
        
        // Base zonal flow varies with latitude
        scalar_t u_base = 5.0f * std::sin(M_PI * y_norm);
        
        for (index_t x = 0; x < grid_width; ++x) {
            // Normalized x-coordinate
            scalar_t x_norm = static_cast<scalar_t>(x) / (grid_width - 1);
            
            // Wave perturbation
            scalar_t wave_phase = wave_k * x - 0.1f * y_norm;
            scalar_t wave_amp = amplitude * std::exp(-std::pow(y_norm - 0.5f, 2) / 0.05f);
            
            // Velocity perturbations
            scalar_t u = u_base + wave_amp * std::sin(wave_phase);
            scalar_t v = wave_amp * std::cos(wave_phase);
            
            // Height field
            scalar_t h = h_mean + wave_amp * std::cos(wave_phase);
            
            velocity.set(x, y, u, v);
            height(x, y) = h;
        }
    }
    
    // Calculate diagnostics
    grid.calculateDiagnostics();
}

// FrontInitialCondition implementation
FrontInitialCondition::FrontInitialCondition(
    scalar_t y_position, scalar_t width, scalar_t temp_difference, scalar_t wind_shear
) {
    setParameter("y_position", y_position);
    setParameter("width", width);
    setParameter("temp_difference", temp_difference);
    setParameter("wind_shear", wind_shear);
}

void FrontInitialCondition::initialize(WeatherGrid& grid) const {
    scalar_t y_position = getParameter("y_position", 0.5f);
    scalar_t width = getParameter("width", 0.05f);
    scalar_t temp_difference = getParameter("temp_difference", 10.0f);
    scalar_t wind_shear = getParameter("wind_shear", 5.0f);
    
    auto& velocity = grid.getVelocityField();
    auto& temperature = grid.getTemperatureField();
    auto& pressure = grid.getPressureField();
    
    index_t grid_width = grid.getWidth();
    index_t grid_height = grid.getHeight();
    
    scalar_t y_pos_grid = y_position * (grid_height - 1);
    scalar_t width_grid = width * grid_height;
    
    // Create a front (sharp temperature gradient)
    for (index_t y = 0; y < grid_height; ++y) {
        // Distance from front center
        scalar_t dy = y - y_pos_grid;
        
        // Temperature transition using tanh function
        scalar_t t_transition = std::tanh(dy / width_grid);
        scalar_t temperature_val = 288.15f + 0.5f * temp_difference * t_transition;
        
        // Wind shear across the front
        scalar_t u = 0.5f * wind_shear * t_transition;
        
        for (index_t x = 0; x < grid_width; ++x) {
            velocity.set(x, y, u, 0.0f);
            temperature(x, y) = temperature_val;
            
            // Pressure adjusts with temperature (simplified - real fronts are more complex)
            scalar_t p = 1013.25f - 0.1f * temp_difference * t_transition;
            pressure(x, y) = p;
        }
    }
    
    // Calculate diagnostics
    grid.calculateDiagnostics();
}

// MountainInitialCondition implementation
MountainInitialCondition::MountainInitialCondition(
    scalar_t x_center, scalar_t y_center, scalar_t radius, scalar_t height, scalar_t u_base
) {
    setParameter("x_center", x_center);
    setParameter("y_center", y_center);
    setParameter("radius", radius);
    setParameter("height", height);
    setParameter("u_base", u_base);
}

void MountainInitialCondition::initialize(WeatherGrid& grid) const {
    scalar_t x_center = getParameter("x_center", 0.3f);
    scalar_t y_center = getParameter("y_center", 0.5f);
    scalar_t radius = getParameter("radius", 0.1f);
    scalar_t mountain_height = getParameter("height", 1.0f);
    scalar_t u_base = getParameter("u_base", 5.0f);
    
    auto& velocity = grid.getVelocityField();
    auto& height = grid.getHeightField();
    
    index_t grid_width = grid.getWidth();
    index_t grid_height = grid.getHeight();
    
    scalar_t x_center_grid = x_center * (grid_width - 1);
    scalar_t y_center_grid = y_center * (grid_height - 1);
    scalar_t radius_grid = radius * std::min(grid_width, grid_height);
    
    // Create mountain and flow around/over it
    for (index_t y = 0; y < grid_height; ++y) {
        for (index_t x = 0; x < grid_width; ++x) {
            // Distance from mountain center
            scalar_t dx = x - x_center_grid;
            scalar_t dy = y - y_center_grid;
            scalar_t r = std::sqrt(dx * dx + dy * dy);
            
            // Mountain shape (bell curve)
            scalar_t mountain_profile = 0.0f;
            if (r <= 2.0f * radius_grid) {
                mountain_profile = mountain_height * 
                    std::exp(-(r * r) / (radius_grid * radius_grid));
            }
            
            // Base height plus mountain
            scalar_t h = 10.0f + mountain_profile;
            
            // Basic flow distortion (simplified)
            scalar_t u = u_base;
            scalar_t v = 0.0f;
            
            if (r <= 3.0f * radius_grid) {
                // Reduce flow over mountain and divert around
                scalar_t flow_reduction = 0.7f * mountain_profile / mountain_height;
                u *= (1.0f - flow_reduction);
                
                // Divert flow around the mountain (very simplified)
                if (r > 0.0f) {
                    v = -0.5f * flow_reduction * u_base * dy / r;
                }
            }
            
            velocity.set(x, y, u, v);
            height(x, y) = h;
        }
    }
    
    // Calculate diagnostics
    grid.calculateDiagnostics();
}

// AtmosphericProfileInitialCondition implementation
AtmosphericProfileInitialCondition::AtmosphericProfileInitialCondition(
    const std::string& profile_name
) : profile_name_(profile_name) {
    setStringParameter("profile_name", profile_name);
    
    // Load the selected profile
    if (profile_name == "standard") {
        loadStandardProfile();
    } else if (profile_name == "tropical") {
        loadTropicalProfile();
    } else if (profile_name == "polar") {
        loadPolarProfile();
    } else {
        // Default to standard
        loadStandardProfile();
    }
}

void AtmosphericProfileInitialCondition::initialize(WeatherGrid& grid) const {
    auto& temperature = grid.getTemperatureField();
    auto& pressure = grid.getPressureField();
    auto& humidity = grid.getHumidityField();
    auto& velocity = grid.getVelocityField();
    
    index_t grid_width = grid.getWidth();
    index_t grid_height = grid.getHeight();
    
    // Applied selected profile with some spatial variation
    for (index_t y = 0; y < grid_height; ++y) {
        // Normalized y coordinate (latitude-like)
        scalar_t y_norm = static_cast<scalar_t>(y) / (grid_height - 1);
        
        // Get profile value based on "latitude"
        scalar_t t_base = 288.15f; // Default temperature
        scalar_t p_base = 1013.25f; // Default pressure
        scalar_t q_base = 0.0f;     // Default humidity
        scalar_t u_base = 0.0f;     // Default u-wind
        scalar_t v_base = 0.0f;     // Default v-wind
        
        // Simple interpolation between profiles
        if (!temperature_profile_.empty()) {
            size_t idx = static_cast<size_t>(y_norm * (temperature_profile_.size() - 1));
            idx = std::min(idx, temperature_profile_.size() - 1);
            t_base = temperature_profile_[idx];
            p_base = pressure_levels_[idx];
            q_base = humidity_profile_[idx];
            u_base = u_wind_profile_[idx];
            v_base = v_wind_profile_[idx];
        }
        
        for (index_t x = 0; x < grid_width; ++x) {
            // Add some longitudinal variation
            scalar_t x_norm = static_cast<scalar_t>(x) / (grid_width - 1);
            
            // Small sinusoidal variations in parameters
            scalar_t t_var = 2.0f * std::sin(2.0f * M_PI * x_norm);
            scalar_t p_var = 2.0f * std::cos(2.0f * M_PI * x_norm);
            scalar_t q_var = 0.02f * std::sin(4.0f * M_PI * x_norm);
            
            temperature(x, y) = t_base + t_var;
            pressure(x, y) = p_base + p_var;
            humidity(x, y) = q_base + q_var;
            velocity.set(x, y, u_base, v_base);
        }
    }
    
    // Calculate diagnostics
    grid.calculateDiagnostics();
}

void AtmosphericProfileInitialCondition::loadStandardProfile() {
    // Standard atmospheric profile
    // Values correspond to different latitudes
    
    // Pressure levels (hPa)
    pressure_levels_ = {1013.0f, 1011.0f, 1009.0f, 1005.0f, 1000.0f, 
                         995.0f,  990.0f,  985.0f,  980.0f,  975.0f};
    
    // Temperature profile (K)
    temperature_profile_ = {298.0f, 295.0f, 292.0f, 288.0f, 285.0f,
                            282.0f, 278.0f, 275.0f, 272.0f, 268.0f};
    
    // Humidity profile (0-1)
    humidity_profile_ = {0.8f, 0.75f, 0.7f, 0.65f, 0.6f,
                         0.55f, 0.5f, 0.45f, 0.4f, 0.35f};
    
    // U-wind profile (m/s)
    u_wind_profile_ = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                       12.0f, 10.0f, 8.0f, 6.0f, 4.0f};
    
    // V-wind profile (m/s)
    v_wind_profile_ = {0.0f, 1.0f, 2.0f, 1.0f, 0.0f,
                      -1.0f,-2.0f,-1.0f, 0.0f, 1.0f};
}

void AtmosphericProfileInitialCondition::loadTropicalProfile() {
    // Tropical atmospheric profile
    
    // Pressure levels (hPa)
    pressure_levels_ = {1010.0f, 1009.0f, 1008.0f, 1007.0f, 1006.0f, 
                        1005.0f, 1004.0f, 1003.0f, 1002.0f, 1001.0f};
    
    // Temperature profile (K)
    temperature_profile_ = {303.0f, 302.0f, 301.0f, 300.0f, 299.0f,
                            298.0f, 297.0f, 296.0f, 295.0f, 294.0f};
    
    // Humidity profile (0-1)
    humidity_profile_ = {0.9f, 0.89f, 0.88f, 0.87f, 0.86f,
                         0.85f, 0.84f, 0.83f, 0.82f, 0.81f};
    
    // U-wind profile (m/s) - Trade winds
    u_wind_profile_ = {-5.0f, -6.0f, -7.0f, -8.0f, -7.0f,
                       -6.0f, -5.0f, -4.0f, -3.0f, -2.0f};
    
    // V-wind profile (m/s)
    v_wind_profile_ = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f,
                        1.0f,  0.5f, 0.0f,-0.5f,-1.0f};
}

void AtmosphericProfileInitialCondition::loadPolarProfile() {
    // Polar atmospheric profile
    
    // Pressure levels (hPa)
    pressure_levels_ = {1020.0f, 1018.0f, 1016.0f, 1014.0f, 1012.0f, 
                         1010.0f, 1008.0f, 1006.0f, 1004.0f, 1002.0f};
    
    // Temperature profile (K)
    temperature_profile_ = {260.0f, 258.0f, 256.0f, 254.0f, 252.0f,
                            250.0f, 248.0f, 246.0f, 244.0f, 242.0f};
    
    // Humidity profile (0-1)
    humidity_profile_ = {0.3f, 0.29f, 0.28f, 0.27f, 0.26f,
                         0.25f, 0.24f, 0.23f, 0.22f, 0.21f};
    
    // U-wind profile (m/s)
    u_wind_profile_ = {10.0f, 12.0f, 14.0f, 16.0f, 18.0f,
                       20.0f, 18.0f, 16.0f, 14.0f, 12.0f};
    
    // V-wind profile (m/s)
    v_wind_profile_ = {0.0f, -1.0f, -2.0f, -3.0f, -4.0f,
                      -3.0f, -2.0f, -1.0f,  0.0f,  1.0f};
}

// Function to register all available initial conditions
void registerAllInitialConditions() {
    auto& factory = InitialConditionFactory::getInstance();
    
    // Register uniform initial condition
    factory.registerInitialCondition("uniform", []() {
        return std::make_shared<UniformInitialCondition>();
    });
    
    // Register random initial condition
    factory.registerInitialCondition("random", []() {
        return std::make_shared<RandomInitialCondition>();
    });
    
    // Register zonal flow initial condition
    factory.registerInitialCondition("zonal_flow", []() {
        return std::make_shared<ZonalFlowInitialCondition>();
    });
    
    // Register vortex initial condition
    factory.registerInitialCondition("vortex", []() {
        return std::make_shared<VortexInitialCondition>();
    });
    
    // Register jet stream initial condition
    factory.registerInitialCondition("jet_stream", []() {
        return std::make_shared<JetStreamInitialCondition>();
    });
    
    // Register breaking wave initial condition
    factory.registerInitialCondition("breaking_wave", []() {
        return std::make_shared<BreakingWaveInitialCondition>();
    });
    
    // Register front initial condition
    factory.registerInitialCondition("front", []() {
        return std::make_shared<FrontInitialCondition>();
    });
    
    // Register mountain initial condition
    factory.registerInitialCondition("mountain", []() {
        return std::make_shared<MountainInitialCondition>();
    });
    
    // Register atmospheric profile initial conditions
    factory.registerInitialCondition("standard_atmosphere", []() {
        return std::make_shared<AtmosphericProfileInitialCondition>("standard");
    });
    
    factory.registerInitialCondition("tropical_atmosphere", []() {
        return std::make_shared<AtmosphericProfileInitialCondition>("tropical");
    });
    
    factory.registerInitialCondition("polar_atmosphere", []() {
        return std::make_shared<AtmosphericProfileInitialCondition>("polar");
    });
}

} // namespace weather_sim