// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

/** Numeric domains that are described by their bounds instead of a list of values. */
namespace alpakaTune
{
    /** A bounded runtime numeric domain, sampled without materializing its range.
     */
    template<typename T>
    requires std::integral<T> || std::floating_point<T>
    class Interval
    {
    public:
        using value_type = T;
        static constexpr bool runtimeDomain = true;
        static constexpr bool enumerable = std::integral<T>;
        /** Unsigned type of at least int width: distances of narrower value types must not be promoted to int. */
        using Distance = typename std::conditional_t<
            std::integral<T>,
            std::make_unsigned<std::common_type_t<T, int>>,
            std::type_identity<T>>::type;

        Interval(T minimum, T maximum, T step = T{1}) : m_minimum(minimum), m_maximum(maximum), m_step(step)
        {
            if(!std::isfinite(static_cast<long double>(minimum)) || !std::isfinite(static_cast<long double>(maximum))
               || !std::isfinite(static_cast<long double>(step)) || maximum < minimum || step <= T{0})
                throw std::invalid_argument{"A numeric domain requires finite bounds, "
                                            "min <= max, and a positive step."};
            if constexpr(std::integral<T>)
            {
                using U = Distance;
                auto const distance = static_cast<U>(maximum) - static_cast<U>(minimum);
                auto const quotient = static_cast<U>(distance / static_cast<U>(step));
                if(quotient >= std::numeric_limits<std::size_t>::max())
                    throw std::overflow_error{"The domain has too many addressable values."};
                m_size = static_cast<std::size_t>(quotient) + 1u;
            }
            else
                m_size = minimum == maximum ? 1u : (1u << 24u) + 1u;
        }

        [[nodiscard]] auto size() const noexcept -> std::size_t
        {
            return m_size;
        }

        [[nodiscard]] auto minimum() const noexcept -> T
        {
            return m_minimum;
        }

        [[nodiscard]] auto maximum() const noexcept -> T
        {
            return m_maximum;
        }

        [[nodiscard]] auto step() const noexcept -> T
        {
            return m_step;
        }

        [[nodiscard]] auto at(std::size_t index) const -> T
        {
            if(index >= size())
                throw std::out_of_range{"Numeric domain index out of range."};
            if constexpr(std::integral<T>)
            {
                using U = Distance;
                return static_cast<T>(static_cast<U>(m_minimum) + static_cast<U>(index) * static_cast<U>(m_step));
            }
            else
            {
                if(size() == 1u)
                    return m_minimum;
                auto const ratio = static_cast<long double>(index) / static_cast<long double>(size() - 1u);
                return static_cast<T>((1.0L - ratio) * m_minimum + ratio * m_maximum);
            }
        }

        [[nodiscard]] auto indexOf(T value) const -> std::optional<std::size_t>
        {
            if(value < m_minimum || value > m_maximum)
                return std::nullopt;
            if constexpr(std::integral<T>)
            {
                using U = Distance;
                auto const offset = static_cast<U>(value) - static_cast<U>(m_minimum);
                if(offset % static_cast<U>(m_step) != 0u)
                    return std::nullopt;
                return static_cast<std::size_t>(offset / static_cast<U>(m_step));
            }
            else
            {
                if(!std::isfinite(value))
                    return std::nullopt;
                if(size() == 1u)
                    return 0u;
                auto const ratio = (static_cast<long double>(value) - m_minimum)
                                   / (static_cast<long double>(m_maximum) - m_minimum);
                return static_cast<std::size_t>(std::llround(ratio * (size() - 1u)));
            }
        }

    private:
        T m_minimum, m_maximum, m_step;
        std::size_t m_size{};
    };

    template<typename T>
    [[nodiscard]] auto interval(T minimum, T maximum, T step = T{1}) -> Interval<T>
    {
        return {minimum, maximum, step};
    }

    /** A child domain constrained by values of named parent parameters. */
    template<typename T_Domain, typename T_Generator, typename... T_Names>
    struct Dependent : T_Domain
    {
        static_assert(
            requires(T_Domain const& domain) {
                typename T_Domain::value_type;
                domain.at(0u);
                domain.size();
            },
            "A dependent child domain must supply runtime values; compiled "
            "parameters can be parents.");
        static constexpr bool runtimeDomain = true;
        using ParentNames = std::tuple<T_Names...>;
        T_Generator generator;

        explicit Dependent(T_Domain values, T_Generator function)
            : T_Domain(std::move(values))
            , generator(std::move(function))
        {
        }
    };

    template<typename T_Domain, typename T_Name, typename T_Generator>
    [[nodiscard]] auto dependent(T_Domain values, T_Name, T_Generator generator)
    {
        return Dependent<T_Domain, T_Generator, T_Name>{std::move(values), std::move(generator)};
    }

    template<typename T_Domain, typename... T_Names, typename T_Generator>
    [[nodiscard]] auto dependent(T_Domain values, std::tuple<T_Names...>, T_Generator generator)
    {
        return Dependent<T_Domain, T_Generator, T_Names...>{std::move(values), std::move(generator)};
    }
} // namespace alpakaTune
