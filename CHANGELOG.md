# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

First release of alpakaTune, a header-only C++20 library for tuning launch shapes and kernel parameters of
[alpaka](https://github.com/alpaka-group/alpaka) kernels. It builds on alpaka 3.0.0.

### Added

- Tuners created with `makeTuner` that select a configuration on every `enqueue` and learn from measured launches.
- Runtime (`RVals`), compile-time (`CVals`, `CTypes`) and automatically generated candidates, including launch
  shapes derived from a `FrameSpec`.
- Exhaustive, random, simulated-annealing, Bayesian-optimization and learned-hybrid search strategies.
- Independent exploration (online, offline) and selection (adaptive, fixed) policies, configured in YAML or in code.
- Persistent JSON histories for replaying or adapting saved measurements in later runs.
- Application-defined metrics, and optional counter-based metrics through alpakaMetrics (experimental).
- Examples, a benchmark runner and visualizer, and a user guide.

### Changed

- The project is named alpakaTune and its dependency alpaka throughout. The FetchContent name of the dependency is
  `alpaka`; a local override is spelled `FETCHCONTENT_SOURCE_DIR_ALPAKA`.
- Identifiers follow alpaka's naming rules, checked by clang-tidy. Template parameters carry the `T_` prefix and member
  type aliases are written in CamelCase, for example `CVals::Values`, `ValuesType` and `EntriesType`.
- C++ sources are formatted with alpaka's clang-format style.

### Fixed

- `RVals(minimum, maximum, step)` over unsigned types no longer continues beyond the maximum when the step exceeds it.
- `generate::linSpace` and `generate::logSpace` end when the following value is not representable instead of looping
  until memory is exhausted.
- `Interval` over 8 and 16 bit integers reports its size and the index of a value correctly.
- A recommendation that reveals a forbidden candidate no longer counts as a strategy retry, so a search reaches the
  permitted candidates behind a long run of forbidden ones. A space without permitted candidates ends with
  `noValidConfiguration`.
- The Bayesian-optimization strategy no longer reports a singular kernel matrix for close observations.
- An offline replay with fixed selection accepts a `runsPerCandidate` above `historyWindowSize`.
- A budgeted tuner accepts the metric of a launch that the application invalidated.
