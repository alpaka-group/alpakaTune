// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/interfaces/Strategy.hpp>
#include <alpakaTune/strategy/SimulatedAnnealingStrategy.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
    using alpakaTune::ConfigurationState;
    using alpakaTune::ParameterConfiguration;
    using alpakaTune::RecommendationDisposition;
    using alpakaTune::RuntimeComparison;
    using alpakaTune::RuntimeObservation;

    [[nodiscard]] auto observation(
        double seconds,
        ConfigurationState state = ConfigurationState::retired,
        RuntimeComparison comparison = RuntimeComparison::inconclusive) -> RuntimeObservation
    {
        return {seconds, 8u, 8u, state, comparison, true};
    }

    /** A context without a catalog whose observations the test controls. */
    class CartesianContext : public alpakaTune::StrategyContext
    {
    public:
        explicit CartesianContext(std::vector<std::size_t> sizes) : m_sizes(std::move(sizes))
        {
        }

        [[nodiscard]] auto parameterSizes() const noexcept -> std::span<std::size_t const> override
        {
            return m_sizes;
        }

        [[nodiscard]] auto runtimeFor(ParameterConfiguration const& configuration) const
            -> std::optional<RuntimeObservation> override
        {
            auto const found = m_observations.find(configuration);
            return found == m_observations.end() ? std::nullopt : std::optional{found->second};
        }

        void record(ParameterConfiguration const& configuration, RuntimeObservation value)
        {
            m_observations.insert_or_assign(configuration, value);
        }

        void resize(std::vector<std::size_t> sizes)
        {
            m_sizes = std::move(sizes);
        }

    private:
        std::vector<std::size_t> m_sizes;
        std::map<ParameterConfiguration, RuntimeObservation> m_observations;
    };

    /** Five one-dimensional catalog candidates at 0, 0.25, 0.5, 0.75 and 1. */
    class CatalogContext final : public CartesianContext
    {
    public:
        CatalogContext() : CartesianContext({5u})
        {
        }

        [[nodiscard]] auto hasCandidateCatalog() const noexcept -> bool override
        {
            return true;
        }

        [[nodiscard]] auto candidateCount() const noexcept -> std::size_t override
        {
            return m_available.size();
        }

        [[nodiscard]] auto candidateConfiguration(std::size_t id) const -> ParameterConfiguration override
        {
            return {static_cast<float>(id) / 4.0f};
        }

        [[nodiscard]] auto candidateAvailable(std::size_t id) const -> bool override
        {
            return m_available.at(id);
        }

        [[nodiscard]] auto candidateObservation(std::size_t id) const -> std::optional<RuntimeObservation> override
        {
            return m_candidateObservations.at(id);
        }

        void setAvailable(std::size_t id, bool available)
        {
            m_available.at(id) = available;
        }

        void recordCandidate(std::size_t id, RuntimeObservation value)
        {
            m_candidateObservations.at(id) = value;
        }

    private:
        std::array<bool, 5u> m_available{true, true, true, true, true};
        std::array<std::optional<RuntimeObservation>, 5u> m_candidateObservations;
    };

    /** A strategy that only implements the mandatory interface. */
    class MinimalStrategy final : public alpakaTune::ParameterStrategy
    {
    public:
        [[nodiscard]] auto recommend(alpakaTune::StrategyContext const& context) -> ParameterConfiguration override
        {
            return ParameterConfiguration(context.parameterSizes().size(), 0.5f);
        }

        void recommendationResult(ParameterConfiguration const& configuration, RecommendationDisposition disposition)
            override
        {
            results.emplace_back(configuration, disposition);
        }

        void configurationInvalidated(ParameterConfiguration const& configuration) override
        {
            invalidated.push_back(configuration);
        }

        std::vector<std::pair<ParameterConfiguration, RecommendationDisposition>> results;
        std::vector<ParameterConfiguration> invalidated;
    };

    [[nodiscard]] auto isNormalized(ParameterConfiguration const& configuration) -> bool
    {
        for(auto const value : configuration)
            if(!std::isfinite(value) || value < 0.0f || value > 1.0f)
                return false;
        return true;
    }

    /** Whether every coordinate lies within @p radius of @p centre. */
    [[nodiscard]] auto isNear(ParameterConfiguration const& configuration, ParameterConfiguration const& centre)
        -> bool
    {
        constexpr auto radius = 0.15f;
        if(configuration.size() != centre.size())
            return false;
        for(std::size_t dimension{}; dimension < centre.size(); ++dimension)
            if(std::abs(configuration[dimension] - centre[dimension]) > radius)
                return false;
        return true;
    }

    /** Admit one configuration, finish it, and let the strategy reconcile it. */
    void finish(
        alpakaTune::SimulatedAnnealingStrategy& strategy,
        CartesianContext& context,
        ParameterConfiguration const& configuration,
        RuntimeObservation value)
    {
        strategy.recommendationResult(configuration, RecommendationDisposition::scheduled);
        context.record(configuration, value);
        static_cast<void>(strategy.recommend(context));
    }

    /** Finish enough statistically slower points to reach the minimum temperature.
     *
     * Slower points never replace the accepted state, so only the proposal
     * radius changes: it shrinks to 0.02 and proposals stay next to that state.
     */
    void cool(alpakaTune::SimulatedAnnealingStrategy& strategy, CartesianContext& context)
    {
        constexpr auto steps = 1000u;
        for(auto step = 0u; step < steps; ++step)
        {
            auto const configuration = ParameterConfiguration{static_cast<float>(step) / steps, 0.0f, 1.0f};
            strategy.recommendationResult(configuration, RecommendationDisposition::scheduled);
            context.record(configuration, observation(100.0, ConfigurationState::retired, RuntimeComparison::slower));
        }
        static_cast<void>(strategy.recommend(context));
    }
} // namespace

TEST_CASE("strategy names round-trip through their YAML spelling", "[strategy]")
{
    using alpakaTune::StrategyKind;
    for(auto const kind :
        {StrategyKind::exhaustive,
         StrategyKind::random,
         StrategyKind::simulatedAnnealing,
         StrategyKind::bayesianOptimization,
         StrategyKind::learnedHybrid})
        CHECK(alpakaTune::strategyFromName(alpakaTune::strategyName(kind)) == kind);
    CHECK(alpakaTune::strategyName(StrategyKind::simulatedAnnealing) == std::string_view{"simulated_annealing"});
    CHECK(alpakaTune::strategyName(static_cast<StrategyKind>(99)) == std::string_view{"unknown"});
    CHECK_THROWS_AS(alpakaTune::strategyFromName("annealing"), std::runtime_error);
    CHECK_THROWS_AS(alpakaTune::strategyFromName(""), std::runtime_error);
}

TEST_CASE("normalized configurations are validated against the tuning shape", "[strategy]")
{
    constexpr auto sizes = std::array<std::size_t, 2u>{4u, 8u};
    CHECK_NOTHROW(alpakaTune::validateParameterConfiguration({0.0f, 1.0f}, sizes));
    CHECK_THROWS_AS(alpakaTune::validateParameterConfiguration({0.5f}, sizes), std::invalid_argument);
    CHECK_THROWS_AS(alpakaTune::validateParameterConfiguration({0.5f, 0.5f, 0.5f}, sizes), std::invalid_argument);
    CHECK_THROWS_AS(alpakaTune::validateParameterConfiguration({-0.1f, 0.5f}, sizes), std::invalid_argument);
    CHECK_THROWS_AS(alpakaTune::validateParameterConfiguration({0.5f, 1.1f}, sizes), std::invalid_argument);
    CHECK_THROWS_AS(
        alpakaTune::validateParameterConfiguration({0.5f, std::numeric_limits<float>::quiet_NaN()}, sizes),
        std::invalid_argument);
    CHECK_THROWS_AS(
        alpakaTune::validateParameterConfiguration({std::numeric_limits<float>::infinity(), 0.5f}, sizes),
        std::invalid_argument);
}

TEST_CASE("only a retired observation counts as finished", "[strategy]")
{
    CHECK(observation(1.0, ConfigurationState::retired).isFinished());
    for(auto const state :
        {ConfigurationState::unmeasured, ConfigurationState::warmingUp, ConfigurationState::measuring})
        CHECK_FALSE(observation(1.0, state).isFinished());
}

TEST_CASE("a context without a catalog reports no candidates", "[strategy]")
{
    auto const context = CartesianContext{{2u, 3u}};
    CHECK_FALSE(context.hasCandidateCatalog());
    CHECK(context.candidateCount() == 0u);
    CHECK(context.candidateConfiguration(0u).empty());
    CHECK_FALSE(context.candidateAvailable(0u));
    CHECK_FALSE(context.candidateValid(0u));
    CHECK_FALSE(context.candidateObservation(0u));
}

TEST_CASE("a strategy without catalog support receives catalog feedback through its Cartesian callbacks", "[strategy]")
{
    auto const context = CartesianContext{{2u, 3u}};
    auto strategy = MinimalStrategy{};
    CHECK_FALSE(strategy.supportsCandidateCatalog());
    CHECK_FALSE(strategy.recommendCandidate(context));

    auto const configuration = strategy.recommend(context);
    strategy.candidateRecommendationResult(7u, configuration, RecommendationDisposition::budgetRejected);
    REQUIRE(strategy.results.size() == 1u);
    CHECK(strategy.results.front().first == configuration);
    CHECK(strategy.results.front().second == RecommendationDisposition::budgetRejected);

    strategy.candidateInvalidated(7u, configuration);
    REQUIRE(strategy.invalidated.size() == 1u);
    CHECK(strategy.invalidated.front() == configuration);
}

TEST_CASE("annealing proposals are reproducible and normalized", "[strategy][annealing]")
{
    auto const context = CartesianContext{{2u, 3u, 4u}};
    auto first = alpakaTune::SimulatedAnnealingStrategy{17u};
    auto second = alpakaTune::SimulatedAnnealingStrategy{17u};
    for(auto run = 0u; run < 16u; ++run)
    {
        auto const proposal = first.recommend(context);
        CHECK(proposal.size() == 3u);
        CHECK(isNormalized(proposal));
        CHECK(proposal == second.recommend(context));
    }
}

TEST_CASE("annealing only learns from admitted and finished proposals", "[strategy][annealing]")
{
    auto const start = ParameterConfiguration{0.2f, 0.2f, 0.2f};
    auto context = CartesianContext{{2u, 3u, 4u}};
    auto untouched = alpakaTune::SimulatedAnnealingStrategy{3u};
    auto rejected = alpakaTune::SimulatedAnnealingStrategy{3u};
    auto unfinished = alpakaTune::SimulatedAnnealingStrategy{3u};
    auto finished = alpakaTune::SimulatedAnnealingStrategy{3u};

    for(auto const disposition :
        {RecommendationDisposition::activeDuplicate,
         RecommendationDisposition::restrictionRejected,
         RecommendationDisposition::userInvalidated,
         RecommendationDisposition::metricUnavailable,
         RecommendationDisposition::revisitRejected,
         RecommendationDisposition::budgetRejected,
         RecommendationDisposition::scoreRejected})
        rejected.recommendationResult(start, disposition);
    finished.recommendationResult(start, RecommendationDisposition::scheduled);
    context.record(start, observation(1.0));

    // A rejected proposal leaves the strategy in its initial state.
    auto const initial = untouched.recommend(context);
    CHECK(rejected.recommend(context) == initial);
    // A finished admitted proposal becomes the state that is perturbed.
    CHECK(finished.recommend(context) != initial);

    // An admitted proposal is ignored until its measurement has ended.
    auto const pending = ParameterConfiguration{0.6f, 0.6f, 0.6f};
    unfinished.recommendationResult(pending, RecommendationDisposition::scheduled);
    CHECK(unfinished.recommend(context) == initial);
    context.record(pending, observation(1.0, ConfigurationState::measuring));
    CHECK(unfinished.recommend(context) == untouched.recommend(context));
    context.record(pending, observation(1.0));
    CHECK(unfinished.recommend(context) != untouched.recommend(context));
}

TEST_CASE("cooled annealing stays at its accepted state and moves only to better points", "[strategy][annealing]")
{
    auto const start = ParameterConfiguration{0.2f, 0.2f, 0.2f};
    auto const better = ParameterConfiguration{0.8f, 0.8f, 0.8f};
    auto const dominated = ParameterConfiguration{0.5f, 0.1f, 0.9f};
    auto const worse = ParameterConfiguration{0.1f, 0.9f, 0.5f};
    auto context = CartesianContext{{2u, 3u, 4u}};
    auto strategy = alpakaTune::SimulatedAnnealingStrategy{11u};
    auto staysNear = [&](ParameterConfiguration const& centre)
    {
        for(auto run = 0u; run < 32u; ++run)
            if(!isNear(strategy.recommend(context), centre))
                return false;
        return true;
    };

    finish(strategy, context, start, observation(1.0));
    cool(strategy, context);
    CHECK(staysNear(start));

    // A faster point is always accepted.
    finish(strategy, context, better, observation(0.5));
    CHECK(staysNear(better));

    // A point the rank test already found slower is never accepted, whatever its estimate.
    finish(strategy, context, dominated, observation(0.1, ConfigurationState::retired, RuntimeComparison::slower));
    CHECK(staysNear(better));

    // At the minimum temperature a much slower point is not accepted either.
    finish(strategy, context, worse, observation(1.0e6));
    CHECK(staysNear(better));

    // A changed tuning shape restarts from the centre of the new space.
    context.resize({2u, 3u});
    CHECK(staysNear({0.5f, 0.5f}));
    context.resize({2u, 3u, 4u});

    // Invalidating the accepted state discards it; the next finished point replaces it.
    strategy.configurationInvalidated(better);
    CHECK(isNormalized(strategy.recommend(context)));
    finish(strategy, context, start, observation(2.0));
    CHECK(staysNear(start));

    // Invalidating a pending point removes it before it can be accepted.
    strategy.recommendationResult(better, RecommendationDisposition::scheduled);
    strategy.configurationInvalidated(better);
    context.record(better, observation(0.1));
    CHECK(staysNear(start));
}

TEST_CASE("catalog annealing selects only available candidates", "[strategy][annealing][catalog]")
{
    auto context = CatalogContext{};
    auto strategy = alpakaTune::SimulatedAnnealingStrategy{5u};
    CHECK(strategy.supportsCandidateCatalog());

    context.setAvailable(2u, false);
    for(auto run = 0u; run < 64u; ++run)
    {
        auto const selected = strategy.recommendCandidate(context);
        REQUIRE(selected);
        CHECK(*selected < context.candidateCount());
        CHECK(*selected != 2u);
    }

    for(std::size_t id{}; id < context.candidateCount(); ++id)
        context.setAvailable(id, id == 3u);
    CHECK(strategy.recommendCandidate(context) == 3u);

    context.setAvailable(3u, false);
    CHECK_FALSE(strategy.recommendCandidate(context));
}

TEST_CASE("cooled catalog annealing follows finished admitted candidates", "[strategy][annealing][catalog]")
{
    auto context = CatalogContext{};
    auto strategy = alpakaTune::SimulatedAnnealingStrategy{23u};
    auto schedule = [&](std::size_t id)
    {
        strategy.candidateRecommendationResult(
            id,
            context.candidateConfiguration(id),
            RecommendationDisposition::scheduled);
    };
    auto alwaysSelects = [&](std::size_t id)
    {
        for(auto run = 0u; run < 32u; ++run)
            if(strategy.recommendCandidate(context) != id)
                return false;
        return true;
    };

    // Repeated equal results keep candidate 1 accepted while the temperature falls to its minimum.
    context.recordCandidate(1u, observation(1.0));
    for(auto step = 0u; step < 1000u; ++step)
        schedule(1u);
    CHECK(alwaysSelects(1u));

    // A rejected candidate is ignored, however fast it is.
    context.recordCandidate(4u, observation(0.1));
    strategy.candidateRecommendationResult(
        4u,
        context.candidateConfiguration(4u),
        RecommendationDisposition::budgetRejected);
    CHECK(alwaysSelects(1u));

    // An admitted candidate is ignored until its measurement has ended.
    context.recordCandidate(3u, observation(0.2, ConfigurationState::measuring));
    schedule(3u);
    CHECK(alwaysSelects(1u));
    context.recordCandidate(3u, observation(0.2));
    CHECK(alwaysSelects(3u));

    // A faster candidate is accepted, a much slower one is not.
    schedule(4u);
    CHECK(alwaysSelects(4u));
    context.recordCandidate(0u, observation(1.0e6));
    schedule(0u);
    CHECK(alwaysSelects(4u));

    // Without an accepted candidate the search restarts from the centre of the space.
    strategy.candidateInvalidated(4u, context.candidateConfiguration(4u));
    context.setAvailable(4u, false);
    CHECK(alwaysSelects(2u));

    // An invalidated pending candidate is never accepted.
    schedule(0u);
    strategy.candidateInvalidated(0u, context.candidateConfiguration(0u));
    context.recordCandidate(0u, observation(0.01));
    CHECK(alwaysSelects(2u));
}
