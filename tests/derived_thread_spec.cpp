// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <alpakaTune/adjust/DerivedThreadSpec.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>

namespace
{
    struct Kernel
    {
        ALPAKA_FN_ACC void operator()(auto const&, auto const&...) const
        {
        }
    };

    [[nodiscard]] auto hostDevice()
    {
        return alpaka::onHost::makeDeviceSelector(
                   alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu})
            .makeDevice(0u);
    }

    /** Whether every component of an alpaka vector is one. */
    [[nodiscard]] auto allOne(auto const& vector) -> bool
    {
        for(std::size_t dimension{}; dimension < ALPAKA_TYPEOF(vector)::dim(); ++dimension)
            if(vector[dimension] != 1u)
                return false;
        return true;
    }
} // namespace

TEST_CASE("the serial executor derives one thread in one block from any frame specification", "[launch]")
{
    auto const device = hostDevice();
    auto const bundle = alpaka::KernelBundle{Kernel{}, 1};

    using Vector1 = alpaka::Vec<std::size_t, 1u>;
    auto const derived1 = alpakaTune::deriveThreadSpec(
        device,
        alpaka::onHost::FrameSpec{Vector1{8u}, Vector1{16u}, alpaka::exec::cpuSerial},
        bundle);
    CHECK(ALPAKA_TYPEOF(derived1.getNumBlocks())::dim() == 1u);
    CHECK(allOne(derived1.getNumBlocks()));
    CHECK(allOne(derived1.getNumThreads()));

    using Vector2 = alpaka::Vec<std::size_t, 2u>;
    auto const derived2 = alpakaTune::deriveThreadSpec(
        device,
        alpaka::onHost::FrameSpec{Vector2{4u, 8u}, Vector2{2u, 16u}, alpaka::exec::cpuSerial},
        bundle);
    CHECK(ALPAKA_TYPEOF(derived2.getNumBlocks())::dim() == 2u);
    CHECK(allOne(derived2.getNumBlocks()));
    CHECK(allOne(derived2.getNumThreads()));
}
