// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

// Regression tests of defects in candidate generation, admission and the proposal strategies.

#include <alpakaTune/alpakaTune.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#if defined(__linux__)
#    include <sys/resource.h>
#    include <unistd.h>
#endif

namespace
{
    inline constexpr auto choice = ALPAKA_TUNE_TUNABLE("bugChoice");

    struct Kernel
    {
        ALPAKA_FN_ACC void operator()(auto const&, alpaka::concepts::IMdSpan auto output, auto value) const
        {
            ++output[0u];
            output[1u] = static_cast<int>(value);
        }
    };

    using Vec = alpaka::Vec<std::size_t, 1u>;

    auto hostDevice()
    {
        return alpaka::onHost::makeDeviceSelector(
                   alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu})
            .makeDevice(0u);
    }

    inline constexpr auto outer = ALPAKA_TUNE_TUNABLE("bugOuter");
    inline constexpr auto inner = ALPAKA_TUNE_TUNABLE("bugInner");

    struct PairKernel
    {
        ALPAKA_FN_ACC void operator()(auto const&, alpaka::concepts::IMdSpan auto output, auto first, auto second)
            const
        {
            output[0u] = static_cast<int>(first);
            output[1u] = static_cast<int>(second);
        }
    };

    /** Turns a runaway candidate generator into a failed allocation instead of exhausting the machine.
     *
     * While an instance lives, the process may map 2 GiB beyond the address space it holds already. The allowance
     * is relative because the address, thread and leak sanitizer runtimes reserve terabytes at start-up, so that an
     * absolute limit fails their next mapping. Without a sanitizer the failed allocation is a std::bad_alloc; with
     * one the runtime aborts the test. The previous limit returns with the destructor, so that later test cases of
     * the process are not limited.
     */
    class AddressSpaceLimit
    {
    public:
        AddressSpaceLimit()
        {
#if defined(__linux__)
            // First field of statm: total program size in pages. It stays zero if procfs is not readable.
            auto heldPages = rlim_t{};
            std::ifstream{"/proc/self/statm"} >> heldPages;
            // Failures throw instead of using REQUIRE: nvcc 13.0 hangs on a Catch2 assertion in this constructor.
            if(getrlimit(RLIMIT_AS, &m_previous) != 0)
                throw std::system_error{errno, std::generic_category(), "getrlimit(RLIMIT_AS)"};
            auto limit = m_previous;
            limit.rlim_cur = std::min(
                m_previous.rlim_cur,
                heldPages * static_cast<rlim_t>(sysconf(_SC_PAGESIZE)) + (rlim_t{2u} << 30u));
            if(setrlimit(RLIMIT_AS, &limit) != 0)
                throw std::system_error{errno, std::generic_category(), "setrlimit(RLIMIT_AS)"};
#endif
        }

        AddressSpaceLimit(AddressSpaceLimit const&) = delete;
        auto operator=(AddressSpaceLimit const&) -> AddressSpaceLimit& = delete;

        ~AddressSpaceLimit()
        {
#if defined(__linux__)
            static_cast<void>(setrlimit(RLIMIT_AS, &m_previous));
#endif
        }

#if defined(__linux__)

    private:
        rlimit m_previous{};
#endif
    };
} // namespace

// A step larger than the maximum must not wrap the unsigned distance to the maximum: the range used to continue
// beyond it, and with a small step until the value type itself wrapped.
TEST_CASE("an unsigned RVals range never exceeds its maximum", "[rvals-unsigned-step]")
{
    auto const values = alpakaTune::RVals<std::uint32_t>(0u, 1u, 0xC000'0000u).values();
    CHECK(values == std::vector<std::uint32_t>{0u});

    auto const wide = alpakaTune::RVals<std::uint64_t>(0u, 1u, std::uint64_t{3u} << 62u).values();
    CHECK(wide == std::vector<std::uint64_t>{0u});

    CHECK(alpakaTune::RVals<std::int32_t>(0, 1, 0x4000'0000).values() == std::vector<std::int32_t>{0});
}

// A following value that is not representable wraps below the last value again; the space must end before it.
TEST_CASE("generated spaces end at a last value near the type maximum", "[generate-wraparound]")
{
    auto const addressSpaceLimit = AddressSpaceLimit{};

    SECTION("logSpace with a power-of-two last value")
    {
        auto values = std::optional<alpakaTune::RVals<std::uint8_t>>{};
        REQUIRE_NOTHROW(values = alpakaTune::generate::logSpace<std::uint8_t>(1u, 128u, 2u));
        CHECK(values->size() == 8u);
    }
    SECTION("logSpace over 32 bit")
    {
        auto values = std::optional<alpakaTune::RVals<std::uint32_t>>{};
        REQUIRE_NOTHROW(values = alpakaTune::generate::logSpace<std::uint32_t>(1u, 0x8000'0000u, 2u));
        CHECK(values->size() == 32u);
    }
    SECTION("linSpace up to the type maximum")
    {
        auto values = std::optional<alpakaTune::RVals<std::uint8_t>>{};
        REQUIRE_NOTHROW(values = alpakaTune::generate::linSpace<std::uint8_t>(0u, 255u, 5u));
        CHECK(values->size() == 52u);
    }
}

// Distances of value types narrower than int must wrap in an unsigned type instead of being promoted to int.
TEST_CASE("intervals of 8 and 16 bit integers address their values", "[interval-narrow-integers]")
{
    auto const shorts = alpakaTune::interval<std::int16_t>(-10, 10, 2);
    CHECK(shorts.size() == 11u);
    CHECK(shorts.indexOf(std::int16_t{2}) == std::optional<std::size_t>{6u});

    auto const unitStep = alpakaTune::interval<std::int16_t>(-10, 10);
    REQUIRE(unitStep.size() == 21u);
    CHECK(unitStep.indexOf(std::int16_t{2}) == std::optional<std::size_t>{12u});
    auto roundTrips = std::size_t{};
    for(std::size_t index = 0u; index < unitStep.size(); ++index)
        roundTrips += unitStep.indexOf(unitStep.at(index)) == std::optional<std::size_t>{index} ? 1u : 0u;
    CHECK(roundTrips == unitStep.size());

    auto const bytes = alpakaTune::interval<std::int8_t>(-128, 127, 2);
    CHECK(bytes.size() == 128u);

    auto const ints = alpakaTune::interval<std::int32_t>(-10, 10, 2);
    CHECK(ints.size() == 11u);
    CHECK(ints.indexOf(2) == std::optional<std::size_t>{6u});

    // A preferred hint resolves through indexOf.
    auto const hinted = alpakaTune::autoCandidates(unitStep, alpakaTune::hint::preferred(std::int16_t{2}));
    CHECK(hinted.preferredIndex() == std::optional<std::size_t>{12u});
}

// validate() limits runsPerCandidate to historyWindowSize only for online exploration. An offline fixed replay
// that keeps the measurement policy of its collection run must not fail on that limit at its first enqueue.
TEST_CASE("an offline fixed tuner accepts the validated measurement policy", "[offline-history-window]")
{
    auto cfg = alpakaTune::TunerConfig{};
    cfg.exploration = alpakaTune::ExplorationPolicy::offline;
    cfg.selection = alpakaTune::SelectionPolicy::fixed;
    cfg.runsPerCandidate = 20u;
    cfg.historyWindowSize = 10u;
    cfg.history = {.read = false, .write = false};
    cfg.completeHistory = {.read = false, .write = false};
    REQUIRE_NOTHROW(cfg.validate());

    auto dev = hostDevice();
    auto queue = dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
    auto host = alpaka::onHost::allocHost<int>(Vec{2u});
    auto output = alpaka::onHost::allocLike(dev, host);
    auto frame = alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
    auto bundle = alpaka::KernelBundle{Kernel{}, output.getMdSpan(), alpakaTune::markTunable(choice)};
    auto tuner = alpakaTune::makeTuner(cfg, alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1})}, dev);
    // No history exists here, so the documented outcome is the missing-history error.
    CHECK_THROWS_WITH(
        tuner.enqueue(queue, frame, bundle),
        Catch::Matchers::ContainsSubstring("Offline tuning requires a compatible history"));
    alpaka::onHost::wait(queue);
}

// A proposal that reveals an illegal candidate is not a strategy retry: a run of more illegal tuples than
// maximumConsecutiveStrategyRetries must not end exploration while legal candidates remain behind it.
// makeFrameSpecTuning(frameSpec) produces such runs for a 1D FrameSpec with 2^20 frames.
TEST_CASE("an exhaustive fixed tuner measures every legal candidate", "[exhaustive-restricted-space]")
{
    auto cfg = alpakaTune::TunerConfig{};
    cfg.selection = alpakaTune::SelectionPolicy::fixed;
    cfg.strategy = alpakaTune::StrategyKind::exhaustive;
    cfg.maximumRetiredConfigurations = 1000u;
    cfg.history = {.read = false, .write = false};
    cfg.completeHistory = {.read = false, .write = false};
    REQUIRE(cfg.maximumConsecutiveStrategyRetries == 20u);

    auto dev = hostDevice();
    auto queue = dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
    auto host = alpaka::onHost::allocHost<int>(Vec{2u});
    auto output = alpaka::onHost::allocLike(dev, host);
    auto frame = alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
    auto bundle = alpaka::KernelBundle{
        PairKernel{},
        output.getMdSpan(),
        alpakaTune::markTunable(outer),
        alpakaTune::markTunable(inner)};
    // Legal tuples: outer in {0, 1} with inner in [0, 5). The 25 illegal tuples after (0, 4) precede (1, 0).
    auto tuner = alpakaTune::makeTuner(
        cfg,
        alpakaTune::constrain(
            alpakaTune::TunableBundle{outer(alpakaTune::RVals{0, 1}), inner(alpakaTune::RVals<int>(0, 29))},
            alpakaTune::restrict(inner, [](int value) { return value < 5; })),
        dev);
    for(std::size_t launch = 0u; launch < 200u && !tuner.completed(); ++launch)
        tuner.enqueue(queue, frame, bundle);
    alpaka::onHost::wait(queue);

    REQUIRE(tuner.completed());
    auto const info = tuner.info();
    CHECK(info.candidateCount == 60u);
    CHECK(info.measuredCandidateCount == 10u);
    CHECK(tuner.completionReason() == alpakaTune::TunerCompletionReason::allConfigurations);
}

// An application may invalidate a launch and still submit its metric, which leaves a budgeted tuner without an
// incumbent. This target defines _GLIBCXX_ASSERTIONS, so libstdc++ aborts if the empty incumbent is read.
TEST_CASE("a budgeted tuner accepts the metric of an invalidated first launch", "[budget-invalidated-metric]")
{
    auto cfg = alpakaTune::TunerConfig{};
    cfg.budget = alpakaTune::BudgetConfig{.expectedLaunches = 10000u};
    cfg.history = {.read = false, .write = false};
    cfg.completeHistory = {.read = false, .write = false};

    auto dev = hostDevice();
    auto queue = dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::disabled);
    auto host = alpaka::onHost::allocHost<int>(Vec{2u});
    auto output = alpaka::onHost::allocLike(dev, host);
    host.getMdSpan()[0u] = 0;
    alpaka::onHost::memcpy(queue, output, host);
    auto frame = alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
    auto bundle = alpaka::KernelBundle{Kernel{}, output.getMdSpan(), alpakaTune::markTunable(choice)};
    auto tuner = alpakaTune::makeTuner(
        cfg,
        alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1, 2})},
        dev,
        alpakaTune::elapsedTimeMetric("step_seconds"));

    auto const observed = tuner.enqueueObserved(queue, frame, bundle);
    REQUIRE(observed.purpose == alpakaTune::LaunchPurpose::experiment);
    tuner.lastConfig().valid = false;
    tuner.provideMetric(1.0);
    alpaka::onHost::wait(queue);
    SUCCEED("the metric of the invalidated launch was accepted");
}

// Close observations bring a pivot of the regularized kernel matrix down to its noise bound, where rounding used
// to classify the matrix as singular and the exception left enqueue().
TEST_CASE("the Bayesian strategy keeps proposing over close observations", "[bayesian-kernel-matrix]")
{
    auto cfg = alpakaTune::TunerConfig{};
    cfg.selection = alpakaTune::SelectionPolicy::adaptive;
    cfg.strategy = alpakaTune::StrategyKind::bayesianOptimization;
    cfg.history = {.read = false, .write = false};
    cfg.completeHistory = {.read = false, .write = false};

    auto dev = hostDevice();
    auto queue = dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::disabled);
    auto host = alpaka::onHost::allocHost<int>(Vec{2u});
    auto output = alpaka::onHost::allocLike(dev, host);
    auto frame = alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
    auto bundle = alpaka::KernelBundle{Kernel{}, output.getMdSpan(), alpakaTune::markTunable(choice)};
    auto const values = alpakaTune::RVals<int>(0, 6);
    auto tuner = alpakaTune::makeTuner(
        cfg,
        alpakaTune::TunableBundle{choice(values)},
        dev,
        alpakaTune::customMetric("distance"));
    for(std::size_t launch = 0u; launch < 200u; ++launch)
    {
        auto observed = std::optional<alpakaTune::LaunchObservation>{};
        REQUIRE_NOTHROW(observed = tuner.enqueueObserved(queue, frame, bundle));
        auto const value = values.at(observed->candidateIndex);
        tuner.provideMetric(1.0 + (value > 3 ? value - 3 : 3 - value));
    }
    alpaka::onHost::wait(queue);
    CHECK(tuner.info().bestCandidateIndex == std::optional<std::size_t>{3u});
}
