// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/core/TunerConfig.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    using alpakaTune::Metric;
    using alpakaTune::TunerConfig;
    using Catch::Matchers::ContainsSubstring;

    constexpr auto minimalYaml = "schema_version: 4\ntuning:\n  runs_per_candidate: 1\n";

    [[nodiscard]] auto load(std::string_view yaml) -> TunerConfig
    {
        auto const path = std::filesystem::temp_directory_path() / "alpakaTune-tuner-config-test.yaml";
        {
            std::ofstream output{path};
            output << yaml;
        }
        return TunerConfig::fromYaml(path);
    }

    struct InvalidCase
    {
        std::string message;
        std::function<void(TunerConfig&)> change;
    };

    struct InvalidYaml
    {
        std::string message;
        std::string yaml;
    };
} // namespace

TEST_CASE("metric names round-trip and accept dashes", "[config]")
{
    for(auto const metric :
        {Metric::runtime, Metric::elapsedTime, Metric::instructions, Metric::l2Misses, Metric::weighted})
        CHECK(alpakaTune::metricFromName(alpakaTune::metricName(metric)) == metric);
    CHECK(alpakaTune::metricFromName("l2-misses") == Metric::l2Misses);
    CHECK(alpakaTune::metricFromName("elapsed-time") == Metric::elapsedTime);
    CHECK(alpakaTune::metricName(static_cast<Metric>(99)) == std::string_view{"unknown"});
    CHECK_THROWS_WITH(alpakaTune::metricFromName("energy"), ContainsSubstring("Unknown tuning metric: energy"));

    CHECK(alpakaTune::measuresSeconds(Metric::runtime));
    CHECK(alpakaTune::measuresSeconds(Metric::elapsedTime));
    CHECK_FALSE(alpakaTune::measuresSeconds(Metric::instructions));
    CHECK_FALSE(alpakaTune::measuresSeconds(Metric::l2Misses));
    CHECK_FALSE(alpakaTune::measuresSeconds(Metric::weighted));
}

TEST_CASE("the metric identity includes everything that changes a weighted score", "[config]")
{
    auto config = TunerConfig{};
    CHECK(alpakaTune::metricIdentity(config) == "metric_v1;runtime");
    // Weighted settings are not part of the identity of another metric.
    config.weightedMetric.timeScale = 2.0;
    CHECK(alpakaTune::metricIdentity(config) == "metric_v1;runtime");

    config.metric = Metric::weighted;
    config.weightedMetric.counterScale = 4.0;
    CHECK(alpakaTune::metricIdentity(config) == "metric_v1;weighted;counter=instructions;weights=0.5,0.5;scales=2,4");
    auto const identity = alpakaTune::metricIdentity(config);
    config.weightedMetric.counter = Metric::l2Misses;
    CHECK(alpakaTune::metricIdentity(config) != identity);
    config.weightedMetric.counter = Metric::instructions;
    config.weightedMetric.counterWeight = 0.25;
    CHECK(alpakaTune::metricIdentity(config) != identity);
    config.weightedMetric.counterWeight = 0.5;
    config.weightedMetric.counterScale = 8.0;
    CHECK(alpakaTune::metricIdentity(config) != identity);
}

TEST_CASE("the default configuration and its policy variants are valid", "[config]")
{
    CHECK_NOTHROW(TunerConfig{}.validate());

    auto fixed = TunerConfig{};
    fixed.selection = alpakaTune::SelectionPolicy::fixed;
    fixed.maximumRetiredConfigurations = 3u;
    CHECK_NOTHROW(fixed.validate());
    fixed.maximumRetiredConfigurations.reset();
    fixed.budget = alpakaTune::BudgetConfig{.expectedLaunches = 10u};
    CHECK_NOTHROW(fixed.validate());

    auto offline = TunerConfig{};
    offline.exploration = alpakaTune::ExplorationPolicy::offline;
    offline.selection = alpakaTune::SelectionPolicy::fixed;
    offline.replayFastPath = true;
    CHECK_NOTHROW(offline.validate());

    auto adaptive = TunerConfig{};
    adaptive.horizon = 100u;
    CHECK_NOTHROW(adaptive.validate());

    // A disabled queue keeps its saved parameters without checking them.
    auto disabledQueue = TunerConfig{};
    disabledQueue.queue = alpakaTune::QueueConfig{.disable = true, .maxConsecutiveRuns = 0u};
    CHECK_NOTHROW(disabledQueue.validate());
}

TEST_CASE("validation names the violated invariant", "[config]")
{
    constexpr auto notANumber = std::numeric_limits<double>::quiet_NaN();
    constexpr auto infinity = std::numeric_limits<double>::infinity();
    using alpakaTune::ExplorationPolicy;
    using alpakaTune::SelectionPolicy;
    auto const cases = std::vector<InvalidCase>{
        {"Invalid candidate-space budget", [](TunerConfig& c) { c.space.initialCandidates = 0u; }},
        {"Invalid candidate-space budget", [](TunerConfig& c) { c.space.plateauPatience = 0u; }},
        {"Invalid candidate-space budget", [](TunerConfig& c) { c.space.minimumRelativeImprovement = 1.0; }},
        {"TunerConfig::runsPerCandidate must be greater than zero", [](TunerConfig& c) { c.runsPerCandidate = 0u; }},
        {"TunerConfig::minimumRunsPerCandidate must be greater than zero",
         [](TunerConfig& c) { c.minimumRunsPerCandidate = 0u; }},
        {"must not exceed runsPerCandidate", [](TunerConfig& c) { c.minimumRunsPerCandidate = 2u; }},
        {"TunerConfig::ciCheckInterval", [](TunerConfig& c) { c.ciCheckInterval = 0u; }},
        {"TunerConfig::ciZScore must be positive and finite", [](TunerConfig& c) { c.ciZScore = 0.0; }},
        {"TunerConfig::ciRelativeWidth must be positive and finite",
         [=](TunerConfig& c) { c.ciRelativeWidth = infinity; }},
        {"TunerConfig::outlierMadScale", [](TunerConfig& c) { c.outlierMadScale = -1.0; }},
        {"TunerConfig::mannWhitneyMinimumSamples", [](TunerConfig& c) { c.mannWhitneyMinimumSamples = 0u; }},
        {"TunerConfig::mannWhitneyAlpha must be finite and in (0, 1)",
         [](TunerConfig& c) { c.mannWhitneyAlpha = 1.0; }},
        {"TunerConfig::mannWhitneyAlpha must be finite and in (0, 1)",
         [=](TunerConfig& c) { c.mannWhitneyAlpha = notANumber; }},
        {"TunerConfig::queue.noiseCancellationWindow",
         [](TunerConfig& c) { c.queue = alpakaTune::QueueConfig{.noiseCancellationWindow = 0u}; }},
        {"TunerConfig::queue.maxConsecutiveRuns must be greater than zero",
         [](TunerConfig& c) { c.queue = alpakaTune::QueueConfig{.maxConsecutiveRuns = 0u}; }},
        {"must exceed queue.warmupRuns",
         [](TunerConfig& c) { c.queue = alpakaTune::QueueConfig{.warmupRuns = 3u, .maxConsecutiveRuns = 3u}; }},
        {"TunerConfig::maximumConsecutiveStrategyRetries",
         [](TunerConfig& c) { c.maximumConsecutiveStrategyRetries = 0u; }},
        {"TunerConfig::historyWindowSize", [](TunerConfig& c) { c.historyWindowSize = 0u; }},
        {"TunerConfig::revisitAdmissionSteepness", [](TunerConfig& c) { c.revisitAdmissionSteepness = 0.0; }},
        {"TunerConfig::scoreTemperatureStart", [](TunerConfig& c) { c.scoreTemperatureStart = 0.0; }},
        {"TunerConfig::scoreTemperatureEnd must be positive", [](TunerConfig& c) { c.scoreTemperatureEnd = -1.0; }},
        {"must not exceed scoreTemperatureStart", [](TunerConfig& c) { c.scoreTemperatureEnd = 1.0; }},
        {"TunerConfig::horizonOffsetWithActiveHistory",
         [](TunerConfig& c) { c.horizonOffsetWithActiveHistory = 1.5; }},
        {"TunerConfig::maximumExecutions", [](TunerConfig& c) { c.maximumExecutions = 0u; }},
        {"TunerConfig::maximumRetiredConfigurations", [](TunerConfig& c) { c.maximumRetiredConfigurations = 0u; }},
        {"TunerConfig::adaptiveProbeInterval", [](TunerConfig& c) { c.adaptiveProbeInterval = 0u; }},
        {"Online fixed selection requires", [](TunerConfig& c) { c.selection = SelectionPolicy::fixed; }},
        {"replayFastPath requires fixed selection", [](TunerConfig& c) { c.replayFastPath = true; }},
        {"TunerConfig::horizon must be greater than zero", [](TunerConfig& c) { c.horizon = 0u; }},
        {"horizon requires online",
         [](TunerConfig& c)
         {
             c.selection = SelectionPolicy::fixed;
             c.maximumExecutions = 5u;
             c.horizon = 10u;
         }},
        {"completion guards require online exploration",
         [](TunerConfig& c)
         {
             c.exploration = ExplorationPolicy::offline;
             c.maximumExecutions = 5u;
         }},
        {"must not exceed historyWindowSize",
         [](TunerConfig& c)
         {
             c.selection = SelectionPolicy::fixed;
             c.maximumExecutions = 5u;
             c.runsPerCandidate = 20u;
         }},
        {"TunerConfig::history.file must contain", [](TunerConfig& c) { c.history.file = ""; }},
        {"TunerConfig::completeHistory.file must contain", [](TunerConfig& c) { c.completeHistory.file = ""; }},
        {"TunerConfig::history.sampleCount", [](TunerConfig& c) { c.history.sampleCount = 0u; }},
        {"must use different files",
         [](TunerConfig& c)
         {
             c.history.file = "history.json";
             c.completeHistory.file = "./history.json";
         }},
        {"TunerConfig::learnedFallback",
         [](TunerConfig& c) { c.learnedFallback = alpakaTune::StrategyKind::exhaustive; }},
        {"TunerConfig::learnedCandidatePoolSize must be greater than zero",
         [](TunerConfig& c) { c.learnedCandidatePoolSize = 0u; }},
        {"TunerConfig::learnedCandidateBatchSize must be greater than zero",
         [](TunerConfig& c) { c.learnedCandidateBatchSize = 0u; }},
        {"must not exceed learnedCandidatePoolSize",
         [](TunerConfig& c)
         {
             c.learnedCandidatePoolSize = 4u;
             c.learnedCandidateBatchSize = 8u;
         }},
        {"Invalid application-runtime budget", [](TunerConfig& c) { c.budget = alpakaTune::BudgetConfig{}; }},
        {"Invalid application-runtime budget",
         [](TunerConfig& c)
         { c.budget = alpakaTune::BudgetConfig{.expectedLaunches = 10u, .paybackMultiplier = 0.5}; }},
        {"Invalid application-runtime budget",
         [](TunerConfig& c)
         { c.budget = alpakaTune::BudgetConfig{.expectedLaunches = 10u, .maximumOverheadFraction = 1.0}; }},
        {"weightedMetric.counter must be instructions or",
         [](TunerConfig& c)
         {
             c.metric = Metric::weighted;
             c.weightedMetric.counter = Metric::runtime;
         }},
        {"weights must be finite, non-negative, and not both",
         [](TunerConfig& c)
         {
             c.metric = Metric::weighted;
             c.weightedMetric.timeWeight = 0.0;
             c.weightedMetric.counterWeight = 0.0;
         }},
        {"scales must be set, positive and finite", [](TunerConfig& c) { c.metric = Metric::weighted; }},
        {"scales must be set, positive and finite",
         [](TunerConfig& c)
         {
             c.metric = Metric::weighted;
             c.weightedMetric.timeScale = 1.0;
             c.weightedMetric.counterScale = -1.0;
         }},
#ifndef ALPAKA_TUNE_METRICS
        {"requires linking alpakaTune::metrics", [](TunerConfig& c) { c.metric = Metric::instructions; }},
#endif
    };
    for(auto const& invalid : cases)
    {
        INFO(invalid.message);
        auto config = TunerConfig{};
        invalid.change(config);
        CHECK_THROWS_WITH(config.validate(), ContainsSubstring(invalid.message));
    }
}

TEST_CASE("a complete YAML document sets every policy field", "[config][yaml]")
{
    auto const config = load(R"(schema_version: 4
tuning:
  exploration: online
  selection: adaptive
  adaptive_probe_interval: 6
  strategy: simulated_annealing
  random_seed: nondeterministic
  runs_per_candidate: 4
  minimum_runs_per_candidate: 2
  ci_check_interval: 3
  ci_z_score: 1.96
  ci_relative_width: 0.1
  outlier_mad_scale: 2.5
  mann_whitney_early_stop: false
  mann_whitney_min_samples: 5
  mann_whitney_alpha: 0.01
  maximum_consecutive_strategy_retries: 7
  maximum_executions: null
  maximum_retired_configurations: null
  horizon: 500
  horizon_offset_with_active_history: 0.25
  history_window_size: 12
  revisit_admission_steepness: 8.0
  score_temperature_start: 0.5
  score_temperature_end: 0.125
  weighted_metric:
    counter: l2-misses
    time_weight: 0.25
    counter_weight: 0.75
    time_scale_seconds: 0.5
    counter_scale: 1000
  budget:
    expected_launches: 1000
    maximum_overhead_fraction: 0.25
    payback_multiplier: 3
    initial_expected_improvement: 0.5
    check_interval: 16
    measurement_cost_hint_seconds: 0.5
queue:
  disable: true
  warmup_runs: 2
  noise_cancellation_window: 9
  max_consecutive_runs: 5
history:
  file: compact.json
  read: false
  write: false
  sample_count: 3
complete_history:
  file: complete.json
  read: false
  write: false
learning:
  model: model.atml
  fallback: random
  candidate_pool_size: 64
  candidate_batch_size: 16
space:
  initial_candidates: 4
  refinement_batch_size: 2
  refinement_interval: 3
  maximum_candidates: 50
  maximum_generation_attempts: 60
  plateau_patience: 5
  minimum_relative_improvement: 0.25
)");
    CHECK(config.exploration == alpakaTune::ExplorationPolicy::online);
    CHECK(config.selection == alpakaTune::SelectionPolicy::adaptive);
    CHECK(config.adaptiveProbeInterval == 6u);
    CHECK(config.strategy == alpakaTune::StrategyKind::simulatedAnnealing);
    CHECK_FALSE(config.randomSeed);
    CHECK(config.runsPerCandidate == 4u);
    CHECK(config.minimumRunsPerCandidate == 2u);
    CHECK(config.ciCheckInterval == 3u);
    CHECK(config.ciZScore == 1.96);
    CHECK(config.ciRelativeWidth == 0.1);
    CHECK(config.outlierMadScale == 2.5);
    CHECK_FALSE(config.mannWhitneyEarlyStop);
    CHECK(config.mannWhitneyMinimumSamples == 5u);
    CHECK(config.mannWhitneyAlpha == 0.01);
    CHECK(config.maximumConsecutiveStrategyRetries == 7u);
    CHECK_FALSE(config.maximumExecutions);
    CHECK_FALSE(config.maximumRetiredConfigurations);
    CHECK(config.horizon == 500u);
    CHECK(config.horizonOffsetWithActiveHistory == 0.25);
    CHECK(config.historyWindowSize == 12u);
    CHECK(config.revisitAdmissionSteepness == 8.0);
    CHECK(config.scoreTemperatureStart == 0.5);
    CHECK(config.scoreTemperatureEnd == 0.125);
    CHECK(config.metric == Metric::runtime);
    CHECK(config.weightedMetric.counter == Metric::l2Misses);
    CHECK(config.weightedMetric.timeWeight == 0.25);
    CHECK(config.weightedMetric.counterWeight == 0.75);
    CHECK(config.weightedMetric.timeScale == 0.5);
    CHECK(config.weightedMetric.counterScale == 1000.0);

    REQUIRE(config.budget);
    CHECK(config.budget->expectedLaunches == 1000u);
    CHECK(config.budget->maximumOverheadFraction == 0.25);
    CHECK(config.budget->paybackMultiplier == 3.0);
    CHECK(config.budget->initialExpectedImprovement == 0.5);
    CHECK(config.budget->checkInterval == 16u);
    CHECK(config.budget->measurementCostHintSeconds == 0.5);

    REQUIRE(config.queue);
    CHECK(config.queue->disable);
    CHECK(config.queue->warmupRuns == 2u);
    CHECK(config.queue->noiseCancellationWindow == 9u);
    CHECK(config.queue->maxConsecutiveRuns == 5u);

    CHECK(config.history.file == std::filesystem::path{"compact.json"});
    CHECK_FALSE(config.history.read);
    CHECK_FALSE(config.history.write);
    CHECK(config.history.sampleCount == 3u);
    CHECK(config.completeHistory.file == std::filesystem::path{"complete.json"});
    CHECK_FALSE(config.completeHistory.read);
    CHECK_FALSE(config.completeHistory.write);

    CHECK(config.learnedModelFile == std::filesystem::path{"model.atml"});
    CHECK(config.learnedFallback == alpakaTune::StrategyKind::random);
    CHECK(config.learnedCandidatePoolSize == 64u);
    CHECK(config.learnedCandidateBatchSize == 16u);

    CHECK(config.space.initialCandidates == 4u);
    CHECK(config.space.refinementBatchSize == 2u);
    CHECK(config.space.refinementInterval == 3u);
    CHECK(config.space.maximumCandidates == 50u);
    CHECK(config.space.maximumGenerationAttempts == 60u);
    CHECK(config.space.plateauPatience == 5u);
    CHECK(config.space.minimumRelativeImprovement == 0.25);
}

TEST_CASE("a minimal YAML document keeps the documented defaults", "[config][yaml]")
{
    auto const defaults = TunerConfig{};
    auto const config = load(
        "schema_version: 4\ntuning:\n  runs_per_candidate: 3\n  random_seed: 42\n  horizon: null\n"
        "space:\n  plateau_patience: null\n");
    CHECK(config.runsPerCandidate == 3u);
    // Without an explicit minimum, every configured run is required.
    CHECK(config.minimumRunsPerCandidate == 3u);
    CHECK(config.randomSeed == 42u);
    CHECK_FALSE(config.horizon);
    CHECK_FALSE(config.space.plateauPatience);
    CHECK_FALSE(config.queue);
    CHECK_FALSE(config.budget);
    CHECK_FALSE(config.history.file);
    CHECK_FALSE(config.completeHistory.file);
    CHECK(config.exploration == defaults.exploration);
    CHECK(config.selection == defaults.selection);
    CHECK(config.strategy == alpakaTune::StrategyKind::exhaustive);
    CHECK(config.metric == Metric::runtime);
    CHECK(config.mannWhitneyEarlyStop);
    CHECK(config.historyWindowSize == defaults.historyWindowSize);
    CHECK(config.learnedModelFile == defaults.learnedModelFile);
    CHECK(config.space.maximumCandidates == defaults.space.maximumCandidates);

    auto const path = std::filesystem::temp_directory_path() / "alpakaTune-tuner-config-test.yaml";
    CHECK(alpakaTune::tunerConfig(path).runsPerCandidate == 3u);
    std::filesystem::remove(path);
}

TEST_CASE("unusable YAML paths and documents are rejected with their cause", "[config][yaml]")
{
    CHECK_THROWS_WITH(TunerConfig::fromYaml(""), ContainsSubstring("path must not be empty"));
    CHECK_THROWS_WITH(
        TunerConfig::fromYaml(std::filesystem::temp_directory_path() / "alpakaTune-no-such-config.yaml"),
        ContainsSubstring("does not exist"));

    auto const base = std::string{minimalYaml};
    auto const cases = std::vector<InvalidYaml>{
        {"Unable to load alpakaTune YAML configuration", "tuning: ["},
        {"must contain a map", "- schema_version\n"},
        {"Unknown YAML key in root: bogus", base + "bogus: 1\n"},
        {"Missing YAML key: schema_version", "tuning:\n  runs_per_candidate: 1\n"},
        {"Missing YAML map: tuning", "schema_version: 4\n"},
        {"Missing YAML map: tuning", "schema_version: 4\ntuning: 3\n"},
        {"YAML queue must contain a map", base + "queue: 3\n"},
        {"YAML history must contain a map", base + "history: 3\n"},
        {"YAML complete_history must contain a map", base + "complete_history: 3\n"},
        {"YAML learning must contain a map", base + "learning: 3\n"},
        {"YAML space must contain a map", base + "space: 3\n"},
        {"Unknown YAML key in tuning: bogus", base + "  bogus: 1\n"},
        {"Unknown YAML key in queue: bogus", base + "queue:\n  bogus: 1\n"},
        {"Unknown YAML key in history: bogus", base + "history:\n  bogus: 1\n"},
        {"Unknown YAML key in complete_history: sample_count", base + "complete_history:\n  sample_count: 1\n"},
        {"Unknown YAML key in learning: bogus", base + "learning:\n  bogus: 1\n"},
        {"Unknown YAML key in space: bogus", base + "space:\n  bogus: 1\n"},
        {"Unknown YAML key in tuning.budget: bogus", base + "  budget:\n    bogus: 1\n"},
        {"Unknown YAML key in weighted_metric: bogus", base + "  weighted_metric:\n    bogus: 1\n"},
        {"YAML tuning.budget must contain a map", base + "  budget: 3\n"},
        {"Missing YAML key: expected_launches", base + "  budget: {}\n"},
        {"greater than zero: expected_launches", base + "  budget:\n    expected_launches: 0\n"},
        {"in (0, 1): maximum_overhead_fraction",
         base + "  budget:\n    expected_launches: 10\n    maximum_overhead_fraction: 1.0\n"},
        {"positive and finite: payback_multiplier",
         base + "  budget:\n    expected_launches: 10\n    payback_multiplier: 0\n"},
        {"YAML weighted_metric must contain a map", base + "  weighted_metric: 3\n"},
        {"Unknown tuning metric: energy", base + "  metric: energy\n"},
        {"YAML tuning.strategy must be", base + "  strategy: annealing\n"},
        {"YAML random_seed must be", base + "  random_seed: ~\n"},
        {"YAML random_seed must be", base + "  random_seed: soon\n"},
        {"Missing YAML key: runs_per_candidate", "schema_version: 4\ntuning:\n  strategy: random\n"},
        {"greater than zero: runs_per_candidate", "schema_version: 4\ntuning:\n  runs_per_candidate: 0\n"},
        {"minimum_runs_per_candidate must not exceed", base + "  minimum_runs_per_candidate: 2\n"},
        {"positive and finite: ci_z_score", base + "  ci_z_score: 0\n"},
        {"in (0, 1): mann_whitney_alpha", base + "  mann_whitney_alpha: 1\n"},
        {"horizon_offset_with_active_history requires horizon", base + "  horizon_offset_with_active_history: 0.5\n"},
        {"in [0, 1]: horizon_offset_with_active_history",
         base + "  horizon: 10\n  horizon_offset_with_active_history: 1.5\n"},
        {"YAML history.file must not be empty", base + "history:\n  file: \"\"\n"},
        {"YAML complete_history.file must not be empty", base + "complete_history:\n  file: \"\"\n"},
        {"YAML learning.model must not be empty", base + "learning:\n  model: \"\"\n"},
        {"greater than zero: sample_count", base + "history:\n  sample_count: 0\n"},
        {"greater than zero: plateau_patience", base + "space:\n  plateau_patience: 0\n"},
        {"greater than zero: candidate_pool_size", base + "learning:\n  candidate_pool_size: 0\n"},
        // Cross-field invariants are checked after parsing.
        {"must not exceed learnedCandidatePoolSize",
         base + "learning:\n  candidate_pool_size: 4\n  candidate_batch_size: 8\n"},
        {"Invalid candidate-space budget", base + "space:\n  minimum_relative_improvement: 1.0\n"},
    };
    for(auto const& invalid : cases)
    {
        INFO(invalid.yaml);
        CHECK_THROWS_WITH(load(invalid.yaml), ContainsSubstring(invalid.message));
    }
    std::filesystem::remove(std::filesystem::temp_directory_path() / "alpakaTune-tuner-config-test.yaml");
}
