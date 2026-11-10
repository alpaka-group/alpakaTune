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
