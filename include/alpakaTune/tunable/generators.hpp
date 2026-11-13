// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/tunable/Tunables.hpp"

#include <alpaka/Vec.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace alpakaTune::generate
{

    namespace detail
    {

        template<alpaka::concepts::VectorOrScalar T>
        [[nodiscard]] constexpr auto isPositive(T const& value) -> bool
        {
            if constexpr(alpaka::concepts::Vector<T>)
            {
                for(std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
                    if(value[dimension] <= 0)
                        return false;
                return true;
            }
            else
            {
                return value > 0;
            }
        }

        template<alpaka::concepts::VectorOrScalar T>
        [[nodiscard]] constexpr auto isWithin(T const& value, T const& maximum) -> bool
        {
            if constexpr(alpaka::concepts::Vector<T>)
            {
                for(std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
                    if(value[dimension] > maximum[dimension])
                        return false;
                return true;
            }
            else
            {
                return value <= maximum;
            }
        }

        /** Whether value + step is representable, larger than value and not beyond maximum. */
        template<typename T>
        [[nodiscard]] constexpr auto sumIsWithin(T value, T step, T maximum) -> bool
        {
            if constexpr(std::integral<T>)
                return value <= maximum
                       && static_cast<std::uintmax_t>(maximum) - static_cast<std::uintmax_t>(value)
                              >= static_cast<std::uintmax_t>(step);
            else
                return value + step > value && value + step <= maximum;
        }

        /** Whether value * factor is representable, larger than value and not beyond maximum. */
        template<typename T>
        [[nodiscard]] constexpr auto productIsWithin(T value, T factor, T maximum) -> bool
        {
            if constexpr(std::integral<T>)
                return value <= maximum / factor;
            else
                return value * factor > value && value * factor <= maximum;
        }

        /** Whether every component of the value following in a generated space stays within maximum.
         *
         * The following value is tested before it is formed: it may not be representable, and a wrapped
         * integer would compare as within the maximum again.
         */
        template<alpaka::concepts::VectorOrScalar T>
        [[nodiscard]] constexpr auto nextIsWithin(T const& value, T const& increment, T const& maximum, auto within)
            -> bool
        {
            if constexpr(alpaka::concepts::Vector<T>)
            {
                for(std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
                    if(!within(value[dimension], increment[dimension], maximum[dimension]))
                        return false;
                return true;
            }
            else
            {
                return within(value, increment, maximum);
            }
        }

        template<alpaka::concepts::VectorOrScalar T>
        constexpr void add(T& value, T const& step)
        {
            if constexpr(alpaka::concepts::Vector<T>)
            {
                for(std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
                    value[dimension] += step[dimension];
            }
            else
            {
                value += step;
            }
        }

        template<alpaka::concepts::VectorOrScalar T>
        constexpr void multiply(T& value, T const& factor)
        {
            if constexpr(alpaka::concepts::Vector<T>)
            {
                for(std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
                    value[dimension] *= factor[dimension];
            }
            else
            {
                value *= factor;
            }
        }

    } // namespace detail

    template<alpaka::concepts::VectorOrScalar T>
    [[nodiscard]] auto linSpace(T first, T last, T step) -> RVals<T>
    {
        static_assert(
            !alpaka::isCVector_v<T>,
            "generate::linSpace creates runtime candidates; use CVals or "
            "CTypes for CVec values.");
        if(!detail::isPositive(step))
            throw std::invalid_argument{"generate::linSpace needs a positive step."};
        auto values = std::vector<T>{};
        for(auto value = first; detail::isWithin(value, last); detail::add(value, step))
        {
            values.push_back(value);
            if(!detail::nextIsWithin(
                   value,
                   step,
                   last,
                   [](auto... arguments) { return detail::sumIsWithin(arguments...); }))
                break;
        }
        return RVals<T>{std::move(values)};
    }

    template<alpaka::concepts::VectorOrScalar T>
    [[nodiscard]] auto logSpace(T first, T last, T factor) -> RVals<T>
    {
        static_assert(
            !alpaka::isCVector_v<T>,
            "generate::logSpace creates runtime candidates; use CVals or "
            "CTypes for CVec values.");
        if(!detail::isPositive(first) || !detail::isPositive(factor))
            throw std::invalid_argument{"generate::logSpace needs positive values and factors."};
        if constexpr(alpaka::concepts::Vector<T>)
        {
            for(std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
                if(factor[dimension] <= 1)
                    throw std::invalid_argument{"generate::logSpace needs factors greater than one."};
        }
        else if(factor <= 1)
        {
            throw std::invalid_argument{"generate::logSpace needs a factor greater than one."};
        }
        auto values = std::vector<T>{};
        for(auto value = first; detail::isWithin(value, last); detail::multiply(value, factor))
        {
            values.push_back(value);
            if(!detail::nextIsWithin(
                   value,
                   factor,
                   last,
                   [](auto... arguments) { return detail::productIsWithin(arguments...); }))
                break;
        }
        return RVals<T>{std::move(values)};
    }

    namespace detail
    {
        template<auto T_minimum, auto T_maximum, std::size_t... I>
        consteval auto compileLinear(std::index_sequence<I...>)
        {
            static_assert(T_minimum <= T_maximum);
            return CVals<static_cast<decltype(T_minimum)>(
                T_minimum
                + (static_cast<long double>(T_maximum) - T_minimum) * I
                      / (sizeof...(I) > 1u ? sizeof...(I) - 1u : 1u))...>{};
        }

        template<auto T_minimum, auto T_maximum, auto T_current, auto... T_values>
        consteval auto compileLog()
        {
            static_assert(T_minimum > 0 && T_minimum <= T_maximum);
            if constexpr(T_current >= T_maximum)
                return CVals<T_values..., T_maximum>{};
            else if constexpr(T_current > T_maximum / 2)
                return CVals<T_values..., T_current, T_maximum>{};
            else
                return compileLog<T_minimum, T_maximum, T_current * 2, T_values..., T_current>();
        }
    } // namespace detail
} // namespace alpakaTune::generate

namespace alpakaTune
{
    /** Generate at most eight linearly spaced compiled alternatives from bounds. */
    template<auto T_minimum, auto T_maximum>
    [[nodiscard]] consteval auto compileInterval(hint::Linear = {})
    {
        static_assert(std::integral<decltype(T_minimum)> && std::integral<decltype(T_maximum)>);
        static_assert(T_minimum <= T_maximum);
        constexpr auto count
            = static_cast<std::size_t>(std::min(8.0L, static_cast<long double>(T_maximum) - T_minimum + 1.0L));
        return generate::detail::compileLinear<T_minimum, T_maximum>(std::make_index_sequence<count>{});
    }

    /** Generate powers of two relative to the lower bound, including both bounds.
     */
    template<auto T_minimum, auto T_maximum>
    [[nodiscard]] consteval auto compileInterval(hint::Logarithmic)
    {
        return generate::detail::compileLog<T_minimum, T_maximum, T_minimum>();
    }

    /** Concise automatic compiled alternatives declared using bounds only. */
    template<auto T_minimum, auto T_maximum, typename... T_Hints>
    [[nodiscard]] auto autoCandidates(T_Hints... hints)
    {
        if constexpr((std::same_as<std::remove_cvref_t<T_Hints>, hint::Logarithmic> || ... || false))
            return autoCandidates(compileInterval<T_minimum, T_maximum>(hint::logarithmic), hints...);
        else
            return autoCandidates(compileInterval<T_minimum, T_maximum>(), hints...);
    }
} // namespace alpakaTune
