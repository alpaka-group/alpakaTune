// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

/** Progressive candidate generation from a domain, the set of values a tunable may take, and its hints. */
namespace alpakaTune
{
    namespace hint
    {
        struct Linear
        {
        };

        struct Logarithmic
        {
        };

        struct Categorical
        {
        };

        inline constexpr Linear linear{};
        inline constexpr Logarithmic logarithmic{};
        inline constexpr Categorical categorical{};

        struct Alignment
        {
            std::uint64_t value;
        };

        [[nodiscard]] constexpr auto alignment(std::uint64_t value) -> Alignment
        {
            return {value};
        }

        template<typename T>
        struct Preferred
        {
            T value;
        };

        template<typename T>
        [[nodiscard]] constexpr auto preferred(T value) -> Preferred<T>
        {
            return {value};
        }
    } // namespace hint

    /** Opt a domain into hardware- and feedback-informed candidate generation. */
    template<typename T_Domain>
    class AutoCandidates : public T_Domain
    {
    public:
        using DomainType = T_Domain;
        static constexpr bool automaticCandidates = true;

        explicit AutoCandidates(T_Domain values) : T_Domain(std::move(values))
        {
        }

        void setHint(hint::Linear)
        {
            m_logarithmic = false;
        }

        void setHint(hint::Logarithmic)
        {
            if constexpr(requires { this->minimum(); })
                if(this->minimum() <= 0)
                    throw std::invalid_argument{"Logarithmic domains require positive values."};
            m_logarithmic = true;
        }

        void setHint(hint::Categorical)
        {
            m_categorical = true;
        }

        void setHint(hint::Alignment value)
        {
            if(value.value == 0u)
                throw std::invalid_argument{"Alignment must be positive."};
            m_alignment = value.value;
        }

        template<typename T>
        void setHint(hint::Preferred<T> value)
        {
            if constexpr(requires { this->indexOf(value.value); })
            {
                m_preferred = this->indexOf(value.value);
                if(!m_preferred)
                    throw std::invalid_argument{"Preferred value is outside its domain."};
            }
            else if constexpr(requires { this->values(); })
            {
                auto const& values = this->values();
                for(std::size_t i{}; i < values.size(); ++i)
                    if(values[i] == value.value)
                        m_preferred = i;
                if(!m_preferred)
                    throw std::invalid_argument{"Preferred value is not an alternative."};
            }
            else if constexpr(requires { typename T_Domain::Values; })
            {
                [&]<std::size_t... I>(std::index_sequence<I...>)
                {
                    (
                        [&]
                        {
                            using Value = std::tuple_element_t<I, typename T_Domain::Values>;
                            if constexpr(requires { Value::value == value.value; })
                                if(Value::value == value.value)
                                    m_preferred = I;
                        }(),
                        ...);
                }(std::make_index_sequence<T_Domain::size>{});
                if(!m_preferred)
                    throw std::invalid_argument{"Preferred value is not a compiled alternative."};
            }
            else
                throw std::invalid_argument{"Preferred hint requires numeric alternatives."};
        }

        [[nodiscard]] auto preferredIndex() const -> std::optional<std::size_t>
        {
            return m_preferred;
        }

        [[nodiscard]] auto logarithmic() const -> bool
        {
            return m_logarithmic;
        }

        [[nodiscard]] auto categorical() const -> bool
        {
            return m_categorical;
        }

        [[nodiscard]] auto alignment() const -> std::uint64_t
        {
            return m_alignment;
        }

        [[nodiscard]] auto project(std::size_t index, bool sampleGlobal = false) const -> std::optional<std::size_t>
        {
            if constexpr(requires {
                             this->minimum();
                             this->indexOf(this->at(index));
                         })
            {
                using T = typename T_Domain::value_type;
                auto value = this->at(index);
                if(sampleGlobal && m_logarithmic && this->size() > 1u)
                {
                    auto const fraction = static_cast<long double>(index) / (this->size() - 1u);
                    auto const sampled = std::exp(
                        std::log(static_cast<long double>(this->minimum()))
                        + fraction
                              * (std::log(static_cast<long double>(this->maximum()))
                                 - std::log(static_cast<long double>(this->minimum()))));
                    if constexpr(std::integral<T>)
                    {
                        auto const position = std::round((sampled - this->minimum()) / this->step());
                        index = std::min(this->size() - 1u, static_cast<std::size_t>(std::max(0.0L, position)));
                    }
                    else
                        index = *this->indexOf(static_cast<T>(sampled));
                    value = this->at(index);
                }
                if constexpr(std::integral<T>)
                {
                    if(m_alignment > 1u)
                    {
                        auto aligned = std::floor(static_cast<long double>(value) / m_alignment) * m_alignment;
                        if(aligned < this->minimum())
                            aligned += m_alignment;
                        if(aligned > this->maximum())
                            return std::nullopt;
                        return this->indexOf(static_cast<T>(aligned));
                    }
                }
                else
                    return this->indexOf(value);
            }
            return index;
        }

    private:
        bool m_logarithmic{}, m_categorical{};
        std::uint64_t m_alignment{1u};
        std::optional<std::size_t> m_preferred;
    };

    template<typename T_Domain, typename... T_Hints>
    [[nodiscard]] auto autoCandidates(T_Domain values, T_Hints... hints) -> AutoCandidates<T_Domain>
    {
        auto result = AutoCandidates<T_Domain>{std::move(values)};
        (result.setHint(hints), ...);
        return result;
    }
} // namespace alpakaTune
