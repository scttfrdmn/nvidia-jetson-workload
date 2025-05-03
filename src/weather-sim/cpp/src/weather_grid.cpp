/**
 * @file weather_grid.cpp
 * @brief Implementation of the WeatherGrid class.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include "../include/weather_sim/weather_sim.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace weather_sim {

WeatherGrid::WeatherGrid(index_t width, index_t height, index_t num_levels)
    : width_(width), 
      height_(height), 
      num_levels_(num_levels),
      dx_(1.0),
      dy_(1.0),
      velocity_(width, height),
      height_(width, height),
      pressure_(width, height),
      temperature_(width, height),
      humidity_(width, height),
      vorticity_(width, height),
      divergence_(width, height)
{
    if (width <= 0 || height <= 0 || num_levels <= 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }
    
    reset();
}

WeatherGrid::WeatherGrid(const SimulationConfig& config)
    : width_(config.grid_width),
      height_(config.grid_height),
      num_levels_(config.num_levels),
      dx_(config.dx),
      dy_(config.dy),
      velocity_(config.grid_width, config.grid_height),
      height_(config.grid_width, config.grid_height),
      pressure_(config.grid_width, config.grid_height),
      temperature_(config.grid_width, config.grid_height),
      humidity_(config.grid_width, config.grid_height),
      vorticity_(config.grid_width, config.grid_height),
      divergence_(config.grid_width, config.grid_height)
{
    if (config.grid_width <= 0 || config.grid_height <= 0 || config.num_levels <= 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }
    
    reset();
}

void WeatherGrid::reset() {
    // Reset velocity field to zero
    std::fill(velocity_.u.begin(), velocity_.u.end(), 0.0f);
    std::fill(velocity_.v.begin(), velocity_.v.end(), 0.0f);
    
    // Reset scalar fields to default values
    std::fill(height_.data.begin(), height_.data.end(), 10.0f);  // Default height
    std::fill(pressure_.data.begin(), pressure_.data.end(), 1013.25f);  // Default pressure (hPa)
    std::fill(temperature_.data.begin(), temperature_.data.end(), 288.15f);  // Default temperature (15°C in K)
    std::fill(humidity_.data.begin(), humidity_.data.end(), 0.0f);  // Default humidity (0%)
    
    // Reset diagnostic fields
    std::fill(vorticity_.data.begin(), vorticity_.data.end(), 0.0f);
    std::fill(divergence_.data.begin(), divergence_.data.end(), 0.0f);
}

void WeatherGrid::setSpacing(scalar_t dx, scalar_t dy) {
    if (dx <= 0.0 || dy <= 0.0) {
        throw std::invalid_argument("Grid spacing must be positive");
    }
    
    dx_ = dx;
    dy_ = dy;
}

void WeatherGrid::calculateDiagnostics() {
    const auto& u = velocity_.u;
    const auto& v = velocity_.v;
    
    // Calculate vorticity (curl of velocity field)
    for (index_t y = 0; y < height_; ++y) {
        for (index_t x = 0; x < width_; ++x) {
            // Use central differences with boundary handling
            const index_t left = std::max(0, x - 1);
            const index_t right = std::min(width_ - 1, x + 1);
            const index_t top = std::max(0, y - 1);
            const index_t bottom = std::min(height_ - 1, y + 1);
            
            // Compute partial derivatives
            const scalar_t dv_dx = (v[y * width_ + right] - v[y * width_ + left]) / (2.0f * dx_);
            const scalar_t du_dy = (u[bottom * width_ + x] - u[top * width_ + x]) / (2.0f * dy_);
            
            // Vorticity = dv/dx - du/dy
            vorticity_(x, y) = dv_dx - du_dy;
        }
    }
    
    // Calculate divergence
    for (index_t y = 0; y < height_; ++y) {
        for (index_t x = 0; x < width_; ++x) {
            // Use central differences with boundary handling
            const index_t left = std::max(0, x - 1);
            const index_t right = std::min(width_ - 1, x + 1);
            const index_t top = std::max(0, y - 1);
            const index_t bottom = std::min(height_ - 1, y + 1);
            
            // Compute partial derivatives
            const scalar_t du_dx = (u[y * width_ + right] - u[y * width_ + left]) / (2.0f * dx_);
            const scalar_t dv_dy = (v[bottom * width_ + x] - v[top * width_ + x]) / (2.0f * dy_);
            
            // Divergence = du/dx + dv/dy
            divergence_(x, y) = du_dx + dv_dy;
        }
    }
}

void WeatherGrid::swap(WeatherGrid& other) {
    // Check if dimensions match
    if (width_ != other.width_ || height_ != other.height_ || num_levels_ != other.num_levels_) {
        throw std::invalid_argument("Cannot swap grids of different dimensions");
    }
    
    // Swap velocity fields
    velocity_.u.swap(other.velocity_.u);
    velocity_.v.swap(other.velocity_.v);
    
    // Swap scalar fields
    height_.data.swap(other.height_.data);
    pressure_.data.swap(other.pressure_.data);
    temperature_.data.swap(other.temperature_.data);
    humidity_.data.swap(other.humidity_.data);
    
    // Swap diagnostic fields
    vorticity_.data.swap(other.vorticity_.data);
    divergence_.data.swap(other.divergence_.data);
}

} // namespace weather_sim