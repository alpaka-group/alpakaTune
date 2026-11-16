// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/onHost/Device.hpp>
#include <alpaka/onHost/internal/interface.hpp>

namespace alpakaTune
{

    /**
     * @brief Obtain alpaka's current FrameSpec-to-ThreadSpec mapping.
     *
     * Uses alpaka's internal AdjustThreadSpec extension point.
     * It is provided so a tuning application can inspect or persist the exact
     * launch shape selected for a FrameSpec and kernel bundle. The mapping can be
     * customised by specialising alpaka's corresponding internal operation.
     */
    template<typename T_Device, typename T_FrameSpec, typename T_KernelBundle>
    [[nodiscard]] auto deriveThreadSpec(
        T_Device const& device,
        T_FrameSpec const& frameSpec,
        T_KernelBundle const& kernelBundle)
    {
        return alpaka::onHost::internal::adjustThreadSpec(*device.get(), frameSpec, kernelBundle);
    }

} // namespace alpakaTune
