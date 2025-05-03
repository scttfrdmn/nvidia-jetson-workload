/**
 * @file gpu_adaptability_test.cpp
 * @brief Unit tests for GPU adaptability pattern.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <gtest/gtest.h>
#include <weather_sim/weather_sim.hpp>
#include <weather_sim/gpu_adaptability.hpp>
#include <memory>

namespace weather_sim {
namespace test {

class GPUAdaptabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        AdaptiveKernelManager::getInstance().initialize();
    }
};

// Test device capability detection
TEST_F(GPUAdaptabilityTest, DeviceCapabilityDetection) {
    const auto& capabilities = AdaptiveKernelManager::getInstance().getDeviceCapabilities();
    
    // At minimum, device type should be set
    EXPECT_NE(capabilities.device_type, DeviceType::Unknown);
    
    // Device name should not be empty
    EXPECT_FALSE(capabilities.device_name.empty());
    
    // Summary should produce non-empty output
    EXPECT_FALSE(capabilities.getSummary().empty());
}

// Test kernel parameter selection
TEST_F(GPUAdaptabilityTest, KernelParameterSelection) {
    auto& manager = AdaptiveKernelManager::getInstance();
    
    // Test parameter selection for different operations and sizes
    KernelLaunchParams params1 = manager.getOptimalKernelParams("shallow_water", 256, 256);
    KernelLaunchParams params2 = manager.getOptimalKernelParams("shallow_water", 512, 512);
    
    // Regardless of CUDA availability, should get valid parameters
    EXPECT_GT(params1.block_size_x, 0);
    EXPECT_GT(params1.block_size_y, 0);
    EXPECT_GT(params2.block_size_x, 0);
    EXPECT_GT(params2.block_size_y, 0);
    
    // Grid dimensions should be computed based on problem size
    EXPECT_GT(params1.grid_size_x, 0);
    EXPECT_GT(params1.grid_size_y, 0);
    EXPECT_GT(params2.grid_size_x, 0);
    EXPECT_GT(params2.grid_size_y, 0);
    
    // Parameters might differ for different problem sizes
    // (though not guaranteed, as optimal parameters might be the same)
    bool params_differ = 
        params1.block_size_x != params2.block_size_x ||
        params1.block_size_y != params2.block_size_y ||
        params1.shared_memory_bytes != params2.shared_memory_bytes;
    
    // We won't assert on this, as it depends on the specific implementation
    std::cout << "Parameters differ for different problem sizes: " 
              << (params_differ ? "yes" : "no") << std::endl;
}

// Test GPU workload ratio
TEST_F(GPUAdaptabilityTest, GPUWorkloadRatio) {
    auto& manager = AdaptiveKernelManager::getInstance();
    
    // Get GPU workload ratios for different operations
    float ratio1 = manager.getGpuWorkloadRatio("shallow_water");
    float ratio2 = manager.getGpuWorkloadRatio("barotropic");
    
    // Ratios should be in valid range
    EXPECT_GE(ratio1, 0.0f);
    EXPECT_LE(ratio1, 1.0f);
    EXPECT_GE(ratio2, 0.0f);
    EXPECT_LE(ratio2, 1.0f);
}

// Test backend determination
TEST_F(GPUAdaptabilityTest, DetermineOptimalBackend) {
    auto& manager = AdaptiveKernelManager::getInstance();
    
    // Test backend determination for different problem sizes
    ComputeBackend backend1 = manager.determineOptimalBackend(64, 64, "shallow_water");
    ComputeBackend backend2 = manager.determineOptimalBackend(1024, 1024, "shallow_water");
    
    // Small problems might use CPU, large problems should use GPU or Hybrid
    if (manager.isCudaAvailable()) {
        EXPECT_NE(backend2, ComputeBackend::CPU);
    } else {
        EXPECT_EQ(backend2, ComputeBackend::CPU);
    }
    
    // Output decision
    std::cout << "Backend for small problem (64x64): ";
    switch (backend1) {
        case ComputeBackend::CUDA: std::cout << "CUDA"; break;
        case ComputeBackend::CPU: std::cout << "CPU"; break;
        case ComputeBackend::Hybrid: std::cout << "Hybrid"; break;
        case ComputeBackend::AdaptiveHybrid: std::cout << "AdaptiveHybrid"; break;
    }
    std::cout << std::endl;
    
    std::cout << "Backend for large problem (1024x1024): ";
    switch (backend2) {
        case ComputeBackend::CUDA: std::cout << "CUDA"; break;
        case ComputeBackend::CPU: std::cout << "CPU"; break;
        case ComputeBackend::Hybrid: std::cout << "Hybrid"; break;
        case ComputeBackend::AdaptiveHybrid: std::cout << "AdaptiveHybrid"; break;
    }
    std::cout << std::endl;
}

// Test KernelLaunchParams computeGridDimensions
TEST_F(GPUAdaptabilityTest, ComputeGridDimensions) {
    KernelLaunchParams params;
    params.block_size_x = 16;
    params.block_size_y = 16;
    params.block_size_z = 1;
    
    // Compute grid dimensions for a 100x100 problem
    params.computeGridDimensions(100, 100);
    
    // Grid dimensions should be ceil(problem_size / block_size)
    EXPECT_EQ(params.grid_size_x, 7); // ceil(100/16) = 7
    EXPECT_EQ(params.grid_size_y, 7); // ceil(100/16) = 7
    EXPECT_EQ(params.grid_size_z, 1);
    
    // Test with problem size not divisible by block size
    params.computeGridDimensions(30, 30);
    EXPECT_EQ(params.grid_size_x, 2); // ceil(30/16) = 2
    EXPECT_EQ(params.grid_size_y, 2); // ceil(30/16) = 2
}

// Test performance metrics update and KernelAdapter factory
// This is a basic test since we can't ensure CUDA is available in all test environments
TEST_F(GPUAdaptabilityTest, KernelAdapterFactory) {
    KernelAdapterFactory& factory = KernelAdapterFactory::getInstance();
    
    // Register a test adapter
    auto cpu_adapter = std::make_shared<CPUAdapter>();
    factory.registerAdapter(cpu_adapter);
    
    // Get available adapters
    auto available = factory.getAvailableAdapters();
    EXPECT_GT(available.size(), 0);
    
    // Try to get an adapter
    auto adapter = factory.getAdapter("CPUAdapter");
    EXPECT_NE(adapter, nullptr);
    
    // CPU adapter should always be compatible
    EXPECT_TRUE(adapter->isCompatible());
    EXPECT_EQ(adapter->getName(), "CPUAdapter");
}

// Test hybrid execution manager
TEST_F(GPUAdaptabilityTest, HybridExecutionManager) {
    auto& manager = HybridExecutionManager::getInstance();
    
    // Initialize (should work even if CUDA is not available)
    EXPECT_NO_THROW(manager.initialize());
    
    // Test setting and getting workload ratio
    manager.setGpuWorkloadRatio(0.75f);
    EXPECT_FLOAT_EQ(manager.getGpuWorkloadRatio(), 0.75f);
    
    // Test clamping of invalid ratios
    manager.setGpuWorkloadRatio(1.5f);
    EXPECT_FLOAT_EQ(manager.getGpuWorkloadRatio(), 1.0f);
    
    manager.setGpuWorkloadRatio(-0.5f);
    EXPECT_FLOAT_EQ(manager.getGpuWorkloadRatio(), 0.0f);
    
    // Test workload adjustment
    float initial_ratio = manager.getGpuWorkloadRatio();
    manager.adjustWorkloadRatio(10.0, 10.0); // Equal performance
    EXPECT_FLOAT_EQ(manager.getGpuWorkloadRatio(), initial_ratio); // Should stay the same
    
    // Adjustment with invalid times
    manager.setGpuWorkloadRatio(0.5f);
    manager.adjustWorkloadRatio(-1.0, 10.0);
    EXPECT_FLOAT_EQ(manager.getGpuWorkloadRatio(), 0.5f); // Should not change
}

} // namespace test
} // namespace weather_sim