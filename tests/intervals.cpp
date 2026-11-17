// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/tunable/AutoCandidates.hpp>
#include <alpakaTune/tunable/Intervals.hpp>
#include <alpakaTune/tunable/Tunables.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace
{
    using Catch::Matchers::ContainsSubstring;
    namespace hint = alpakaTune::hint;

    /** A domain that lists its values but cannot look one up. */
    class ListedOnly
    {
    public:
        [[nodiscard]] auto values() const -> std::vector<int> const&
        {
            return m_values;
        }

    private:
        std::vector<int> m_values{3, 5, 7};
    };

    /** A domain without any value access. */
    struct Opaque
    {
    };

    struct FirstParent
    {
    };

    struct SecondParent
    {
    };
} // namespace

static_assert(alpakaTune::Interval<int>::enumerable);
static_assert(!alpakaTune::Interval<double>::enumerable);
static_assert(alpakaTune::Interval<int>::runtimeDomain);

TEST_CASE("integer intervals address exactly their stepped values", "[interval]")
{
    auto const stepped = alpakaTune::interval(-4, 8, 3);
    CHECK(stepped.minimum() == -4);
    CHECK(stepped.maximum() == 8);
    CHECK(stepped.step() == 3);
    REQUIRE(stepped.size() == 5u);
    for(std::size_t index{}; index < stepped.size(); ++index)
    {
        CHECK(stepped.at(index) == -4 + 3 * static_cast<int>(index));
        CHECK(stepped.indexOf(stepped.at(index)) == index);
    }
    CHECK_FALSE(stepped.indexOf(3));
    CHECK_FALSE(stepped.indexOf(-5));
    CHECK_FALSE(stepped.indexOf(9));
    CHECK_THROWS_AS(stepped.at(5u), std::out_of_range);

    // The upper bound is not a value when the step does not reach it.
    auto const truncated = alpakaTune::interval(0, 10, 4);
    REQUIRE(truncated.size() == 3u);
    CHECK(truncated.at(2u) == 8);
    CHECK_FALSE(truncated.indexOf(10));

    CHECK(alpakaTune::interval(7, 7).size() == 1u);
    CHECK(alpakaTune::interval(7, 7).indexOf(7) == 0u);
}

TEST_CASE("floating-point intervals interpolate between their bounds", "[interval]")
{
    auto const unit = alpakaTune::interval(0.0, 1.0);
    auto const last = unit.size() - 1u;
    CHECK(unit.size() == (std::size_t{1u} << 24u) + 1u);
    CHECK(unit.at(0u) == 0.0);
    CHECK(unit.at(last) == 1.0);
    CHECK(unit.at(last / 2u) == 0.5);
    CHECK(unit.indexOf(0.5) == last / 2u);
    CHECK(unit.indexOf(1.0) == last);
    CHECK_FALSE(unit.indexOf(1.5));
    CHECK_FALSE(unit.indexOf(-0.5));
    CHECK_FALSE(unit.indexOf(std::numeric_limits<double>::quiet_NaN()));
    CHECK_THROWS_AS(unit.at(unit.size()), std::out_of_range);

    auto const point = alpakaTune::interval(2.5, 2.5);
    REQUIRE(point.size() == 1u);
    CHECK(point.at(0u) == 2.5);
    CHECK(point.indexOf(2.5) == 0u);
    CHECK_THROWS_AS(point.at(1u), std::out_of_range);
}

TEST_CASE("intervals reject bounds that do not describe a finite range", "[interval]")
{
    constexpr auto infinity = std::numeric_limits<double>::infinity();
    constexpr auto notANumber = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(alpakaTune::interval(8, 1), std::invalid_argument);
    CHECK_THROWS_AS(alpakaTune::interval(1, 8, 0), std::invalid_argument);
    CHECK_THROWS_AS(alpakaTune::interval(1, 8, -1), std::invalid_argument);
    CHECK_THROWS_AS(alpakaTune::interval(0.0, infinity), std::invalid_argument);
    CHECK_THROWS_AS(alpakaTune::interval(-infinity, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(alpakaTune::interval(notANumber, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(alpakaTune::interval(0.0, 1.0, notANumber), std::invalid_argument);
    CHECK_THROWS_AS(alpakaTune::interval(0.0, 1.0, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(
        alpakaTune::interval(std::uint64_t{0u}, std::numeric_limits<std::uint64_t>::max()),
        std::overflow_error);
    CHECK_NOTHROW(
        alpakaTune::interval(std::uint64_t{0u}, std::numeric_limits<std::uint64_t>::max(), std::uint64_t{2u}));
}

TEST_CASE("candidate hints are stored and validated", "[hint]")
{
    auto const plain = alpakaTune::autoCandidates(alpakaTune::interval(1, 8));
    CHECK_FALSE(plain.logarithmic());
    CHECK_FALSE(plain.categorical());
    CHECK(plain.alignment() == 1u);
    CHECK_FALSE(plain.preferredIndex());

    auto const hinted = alpakaTune::autoCandidates(
        alpakaTune::interval(1, 8),
        hint::logarithmic,
        hint::categorical,
        hint::alignment(4u));
    CHECK(hinted.logarithmic());
    CHECK(hinted.categorical());
    CHECK(hinted.alignment() == 4u);

    // A later hint replaces an earlier one.
    CHECK_FALSE(alpakaTune::autoCandidates(alpakaTune::interval(1, 8), hint::logarithmic, hint::linear).logarithmic());
    // Listed values have no lower bound to check.
    CHECK(alpakaTune::autoCandidates(alpakaTune::RVals{-1, 0, 1}, hint::logarithmic).logarithmic());

    CHECK_THROWS_WITH(
        alpakaTune::autoCandidates(alpakaTune::interval(0, 8), hint::logarithmic),
        ContainsSubstring("positive values"));
    CHECK_THROWS_WITH(
        alpakaTune::autoCandidates(alpakaTune::interval(0.0, 1.0), hint::logarithmic),
        ContainsSubstring("positive values"));
    CHECK_THROWS_WITH(
        alpakaTune::autoCandidates(alpakaTune::interval(1, 8), hint::alignment(0u)),
        ContainsSubstring("Alignment must be positive"));
}

TEST_CASE("a preferred value must be one of the alternatives", "[hint]")
{
    CHECK(alpakaTune::autoCandidates(alpakaTune::interval(1, 4), hint::preferred(3)).preferredIndex() == 2u);
    CHECK_THROWS_WITH(
        alpakaTune::autoCandidates(alpakaTune::interval(1, 4), hint::preferred(9)),
        ContainsSubstring("outside its domain"));
    CHECK_THROWS_WITH(
        alpakaTune::autoCandidates(alpakaTune::interval(1, 9, 2), hint::preferred(4)),
        ContainsSubstring("outside its domain"));

    CHECK(alpakaTune::autoCandidates(alpakaTune::RVals{2, 4, 8}, hint::preferred(8)).preferredIndex() == 2u);
    CHECK_THROWS_AS(alpakaTune::autoCandidates(alpakaTune::RVals{2, 4, 8}, hint::preferred(5)), std::invalid_argument);

    CHECK(alpakaTune::autoCandidates(ListedOnly{}, hint::preferred(5)).preferredIndex() == 1u);
    CHECK_THROWS_WITH(
        alpakaTune::autoCandidates(ListedOnly{}, hint::preferred(4)),
        ContainsSubstring("not an alternative"));

    CHECK(alpakaTune::autoCandidates(alpakaTune::CVals<1, 2, 4>{}, hint::preferred(4)).preferredIndex() == 2u);
    CHECK_THROWS_WITH(
        alpakaTune::autoCandidates(alpakaTune::CVals<1, 2, 4>{}, hint::preferred(3)),
        ContainsSubstring("not a compiled alternative"));
    // Types carry no value that a preference could name.
    CHECK_THROWS_WITH(
        alpakaTune::autoCandidates(alpakaTune::CTypes<int, float>{}, hint::preferred(1)),
        ContainsSubstring("not a compiled alternative"));

    CHECK_THROWS_WITH(
        alpakaTune::autoCandidates(Opaque{}, hint::preferred(1)),
        ContainsSubstring("requires numeric alternatives"));
}

TEST_CASE("projection applies alignment and logarithmic sampling", "[hint]")
{
    auto const values = alpakaTune::interval(1, 100);

    auto const plain = alpakaTune::autoCandidates(values);
    CHECK(plain.project(41u) == 41u);
    CHECK(plain.project(41u, true) == 41u);

    auto const aligned = alpakaTune::autoCandidates(values, hint::alignment(32u));
    // 40 rounds down to 32; 5 rounds down to 0, below the domain, and moves up to 32.
    CHECK(aligned.project(*values.indexOf(40)) == values.indexOf(32));
    CHECK(aligned.project(*values.indexOf(5)) == values.indexOf(32));
    CHECK(aligned.project(*values.indexOf(96)) == values.indexOf(96));
    // No multiple of 32 lies in the domain.
    CHECK_FALSE(alpakaTune::autoCandidates(alpakaTune::interval(1, 20), hint::alignment(32u)).project(4u));
    // The aligned value lies between the stepped values.
    CHECK_FALSE(alpakaTune::autoCandidates(alpakaTune::interval(1, 99, 2), hint::alignment(32u)).project(20u));

    auto const powers = alpakaTune::interval(1, 1024);
    auto const logarithmic = alpakaTune::autoCandidates(powers, hint::logarithmic);
    auto const last = powers.size() - 1u;
    // Logarithmic sampling only applies to global samples.
    CHECK(logarithmic.project(last / 2u) == last / 2u);
    CHECK(logarithmic.project(0u, true) == 0u);
    CHECK(logarithmic.project(last, true) == last);
    REQUIRE(logarithmic.project(last / 2u, true));
    CHECK(powers.at(*logarithmic.project(last / 2u, true)) == 32);

    auto const reals = alpakaTune::interval(1.0, 100.0);
    auto const realLast = reals.size() - 1u;
    CHECK(alpakaTune::autoCandidates(reals).project(12345u) == 12345u);
    auto const realLogarithmic = alpakaTune::autoCandidates(reals, hint::logarithmic);
    CHECK(realLogarithmic.project(0u, true) == 0u);
    REQUIRE(realLogarithmic.project(realLast / 2u, true));
    CHECK(std::abs(reals.at(*realLogarithmic.project(realLast / 2u, true)) - 10.0) < 1.0e-4);

    // Listed values are not projected.
    CHECK(alpakaTune::autoCandidates(alpakaTune::RVals{2, 4, 8}, hint::alignment(4u)).project(0u) == 0u);
}

TEST_CASE("a dependent domain keeps its values and names every parent", "[interval]")
{
    auto const child = alpakaTune::dependent(
        alpakaTune::interval(1, 4),
        std::tuple{FirstParent{}, SecondParent{}},
        [](int first, int second) { return alpakaTune::interval(1, first + second); });
    static_assert(std::same_as<decltype(child)::ParentNames, std::tuple<FirstParent, SecondParent>>);
    static_assert(decltype(child)::runtimeDomain);
    CHECK(child.size() == 4u);
    CHECK(child.at(3u) == 4);
    CHECK(child.generator(1, 2).size() == 3u);

    auto const single = alpakaTune::dependent(
        alpakaTune::interval(1, 4),
        FirstParent{},
        [](int parent) { return alpakaTune::interval(1, parent); });
    static_assert(std::same_as<decltype(single)::ParentNames, std::tuple<FirstParent>>);
    CHECK(single.generator(2).size() == 2u);
}
