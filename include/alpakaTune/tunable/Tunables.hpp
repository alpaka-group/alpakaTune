// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/tunable/Domains.hpp"

#include <alpaka/CVec.hpp>
#include <alpaka/Vec.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace alpakaTune
{

    namespace mdPolicy
    {
        /** Tune vector components independently (the default). */
        struct Independent
        {
        };

        /** Tune only complete vectors from the supplied candidate list. */
        struct Listed
        {
        };

        inline constexpr Independent independent{};
        inline constexpr Listed listed{};
    } // namespace mdPolicy

    namespace concepts
    {
        template<typename T>
        concept MdPolicy = std::same_as<std::remove_cvref_t<T>, mdPolicy::Independent>
                           || std::same_as<std::remove_cvref_t<T>, mdPolicy::Listed>;
    } // namespace concepts

    /** A structural string used as a C++20 non-type template parameter. */
    template<std::size_t T_size>
    struct FixedString
    {
        char value[T_size];

        constexpr FixedString(char const (&text)[T_size]) : value{}
        {
            for(std::size_t index = 0u; index < T_size; ++index)
                value[index] = text[index];
        }

        [[nodiscard]] constexpr auto view() const -> std::string_view
        {
            return {value, T_size - 1u};
        }

        constexpr auto operator<=>(FixedString const&) const = default;
    };

    template<FixedString T_name, typename T_Values>
    struct Tunable;

    template<FixedString T_name>
    struct TunableName
    {
        static constexpr auto value = T_name;
        static constexpr auto name = T_name;
        static constexpr bool tunableMarker = true;

        [[nodiscard]] constexpr auto view() const -> std::string_view
        {
            return T_name.view();
        }

        template<typename T_Values>
        [[nodiscard]] constexpr auto operator()(T_Values values) const
            -> Tunable<T_name, std::remove_cvref_t<T_Values>>;

        template<typename T_Values>
        [[nodiscard]] constexpr auto operator()(T_Values values, mdPolicy::Independent) const;

        template<typename T_Values>
        [[nodiscard]] constexpr auto operator()(T_Values values, mdPolicy::Listed) const;
    };

#define ALPAKA_TUNE_TUNABLE(text)                                                                                     \
    ::alpakaTune::TunableName<::alpakaTune::FixedString{text}>                                                        \
    {                                                                                                                 \
    }

// Compatibility spelling for the first frontend draft.
#define ALPAKA_TUNE_NAME(text) ALPAKA_TUNE_TUNABLE(text)

    /** Runtime candidate values. Integral constructors describe inclusive ranges.
     */
    template<typename T>
    class RVals
    {
    public:
        using value_type = T;

        explicit RVals(std::vector<T> values) : m_values(std::move(values))
        {
            validate();
        }

        RVals(std::initializer_list<T> values) : RVals(std::vector<T>{values})
        {
        }

        explicit RVals(T maximum) requires std::integral<T>
            : RVals(T{0}, maximum, T{1})
        {
        }

        RVals(T minimum, T maximum) requires std::integral<T>
            : RVals(minimum, maximum, T{1})
        {
        }

        RVals(T minimum, T maximum, T step) requires std::integral<T>
        {
            if(step <= T{0} || maximum < minimum)
                throw std::invalid_argument{"An RVals range requires min <= max and a positive step."};
            for(auto value = minimum;;)
            {
                m_values.push_back(value);
                if(value > maximum - step)
                    break;
                value = static_cast<T>(value + step);
            }
            validate();
        }

        template<typename... T_Values>
        [[nodiscard]] static auto list(T_Values&&... values) -> RVals
        {
            return RVals{std::vector<T>{static_cast<T>(std::forward<T_Values>(values))...}};
        }

        [[nodiscard]] auto values() const noexcept -> std::vector<T> const&
        {
            return m_values;
        }

        [[nodiscard]] auto size() const noexcept -> std::size_t
        {
            return m_values.size();
        }

        [[nodiscard]] auto at(std::size_t index) const -> typename std::vector<T>::const_reference
        {
            return m_values.at(index);
        }

        [[nodiscard]] auto indexOf(T const& value) const -> std::optional<std::size_t>
        {
            for(std::size_t index{}; index < m_values.size(); ++index)
                if(m_values[index] == value)
                    return index;
            return std::nullopt;
        }

    private:
        void validate() const
        {
            if(m_values.empty())
                throw std::invalid_argument{"RVals requires at least one candidate value."};
        }

        std::vector<T> m_values;
    };

    template<typename T>
    RVals(std::vector<T>) -> RVals<T>;
    template<typename T>
    RVals(std::initializer_list<T>) -> RVals<T>;
    template<std::integral T>
    RVals(T) -> RVals<T>;
    template<std::integral T>
    RVals(T, T) -> RVals<T>;
    template<std::integral T>
    RVals(T, T, T) -> RVals<T>;

    /** Compile-time candidate values. */
    template<auto... T_values>
    struct CVals
    {
        static_assert(sizeof...(T_values) > 0u, "CVals requires at least one candidate value.");
        static constexpr std::size_t size = sizeof...(T_values);
        using Values = std::tuple<std::integral_constant<decltype(T_values), T_values>...>;
    };

    /** Compile-time candidate types, for example differently sized alpaka::CVecs.
     */
    template<typename... T_Values>
    struct CTypes
    {
        static_assert(sizeof...(T_Values) > 0u, "CTypes requires at least one candidate type.");
        static constexpr std::size_t size = sizeof...(T_Values);
        using Values = std::tuple<T_Values...>;
    };

    namespace detail
    {

        template<auto T_current, auto T_maximum, auto T_step, bool T_complete, auto... T_values>
        struct MakeCValsRangeImpl;

        template<auto T_current, auto T_maximum, auto T_step, auto... T_values>
        struct MakeCValsRangeImpl<T_current, T_maximum, T_step, false, T_values...>
        {
            using NextType = decltype(T_current + T_step);
            static constexpr auto next = static_cast<NextType>(T_current + T_step);
            using type =
                typename MakeCValsRangeImpl<next, T_maximum, T_step, (next > T_maximum), T_values..., T_current>::type;
        };

        template<auto T_current, auto T_maximum, auto T_step, auto... T_values>
        struct MakeCValsRangeImpl<T_current, T_maximum, T_step, true, T_values...>
        {
            using type = CVals<T_values...>;
        };

        template<auto T_minimum, auto T_maximum, auto T_step, auto... T_values>
        struct MakeCValsRange
        {
            static_assert(T_step > 0, "CValsRange requires a positive step.");
            static_assert(T_minimum <= T_maximum, "CValsRange requires min <= max.");
            using type =
                typename MakeCValsRangeImpl<T_minimum, T_maximum, T_step, (T_minimum > T_maximum), T_values...>::type;
        };

        template<typename T>
        struct IsRVals : std::false_type
        {
        };

        template<typename T>
        struct IsRVals<RVals<T>> : std::true_type
        {
        };

        /** Adapt an explicit vector list into one categorical tuning dimension. */
        template<typename T_Values>
        struct ListedMdCandidates : T_Values
        {
            static constexpr bool listedMdCandidates = true;

            explicit constexpr ListedMdCandidates(T_Values values) : T_Values(std::move(values))
            {
            }
        };

        template<typename T_Values>
        struct IsRVals<ListedMdCandidates<T_Values>> : IsRVals<T_Values>
        {
        };

        template<typename T_Values>
        inline constexpr bool isListedMdCandidates = requires { T_Values::listedMdCandidates; };

        template<typename T_Domain>
                struct IsRVals<AutoCandidates<T_Domain>> : std::bool_constant < IsRVals<T_Domain>::value
            || requires
        {
            T_Domain::runtimeDomain;
        }>{};

        template<typename T>
        inline constexpr bool isRVals = IsRVals<std::remove_cvref_t<T>>::value;

        template<typename T>
        struct IsCVals : std::false_type
        {
        };

        template<auto... T_values>
        struct IsCVals<CVals<T_values...>> : std::true_type
        {
        };

        template<typename T_Domain>
        struct IsCVals<AutoCandidates<T_Domain>> : IsCVals<T_Domain>
        {
        };

        template<typename T>
        inline constexpr bool isCVals = IsCVals<std::remove_cvref_t<T>>::value;

        template<typename T>
        struct IsCTypes : std::false_type
        {
        };

        template<typename... T_Values>
        struct IsCTypes<CTypes<T_Values...>> : std::true_type
        {
        };

        template<typename T_Values>
        struct IsCTypes<ListedMdCandidates<T_Values>> : IsCTypes<T_Values>
        {
        };

        template<typename T_Domain>
        struct IsCTypes<AutoCandidates<T_Domain>> : IsCTypes<T_Domain>
        {
        };

        template<typename T>
        inline constexpr bool isCTypes = IsCTypes<std::remove_cvref_t<T>>::value;

        template<typename T>
        inline constexpr bool isCompileTimeCandidates = isCVals<T> || isCTypes<T>;

        template<typename T>
        struct IntegerSequenceTraits
        {
            static constexpr bool value = false;
        };

        template<typename T, T... T_values>
        struct IntegerSequenceTraits<std::integer_sequence<T, T_values...>>
        {
            static constexpr bool value = true;
            static constexpr std::size_t dimensionCount = sizeof...(T_values);
            using value_type = T;

            template<std::size_t T_dimension>
            static consteval auto component() -> T
            {
                static_assert(T_dimension < dimensionCount);
                constexpr std::array values{T_values...};
                return values[T_dimension];
            }
        };

        template<typename T>
        inline constexpr bool isIntegerSequence = IntegerSequenceTraits<std::remove_cvref_t<T>>::value;

        template<typename T>
        inline constexpr bool isCompileVector = alpaka::isCVector_v<std::remove_cvref_t<T>> || isIntegerSequence<T>;

        template<typename T>
        consteval auto compileVectorDimensionCount() -> std::size_t
        {
            using Value = std::remove_cvref_t<T>;
            if constexpr(alpaka::isCVector_v<Value>)
                return Value::dim();
            else if constexpr(isIntegerSequence<Value>)
                return IntegerSequenceTraits<Value>::dimensionCount;
            else
                return 1u;
        }

        template<typename T, std::size_t T_dimension>
        consteval auto compileVectorComponent()
        {
            using Value = std::remove_cvref_t<T>;
            if constexpr(alpaka::isCVector_v<Value>)
            {
                static_assert(T_dimension < Value::dim());
                return Value{}[T_dimension];
            }
            else
            {
                return IntegerSequenceTraits<Value>::template component<T_dimension>();
            }
        }

        template<typename T_Values>
        struct CandidateDimensionCount : std::integral_constant<std::size_t, 1u>
        {
        };

        template<typename T, bool = alpaka::isVector_v<T>>
        struct RuntimeCandidateDimensionCount : std::integral_constant<std::size_t, 1u>
        {
        };

        template<typename T>
        struct RuntimeCandidateDimensionCount<T, true>
            : std::integral_constant<std::size_t, static_cast<std::size_t>(T::dim())>
        {
        };

        template<typename T>
        struct CandidateDimensionCount<RVals<T>> : RuntimeCandidateDimensionCount<T>
        {
        };

        template<typename T_First, typename T_Other>
        consteval auto compatibleCompileVector() -> bool
        {
            if constexpr(!isCompileVector<T_Other>)
                return false;
            else
                return compileVectorDimensionCount<T_Other>() == compileVectorDimensionCount<T_First>();
        }

        template<typename T_First, typename... T_Remaining>
        struct CandidateDimensionCount<CTypes<T_First, T_Remaining...>>
            : std::integral_constant<
                  std::size_t,
                  isCompileVector<T_First> ? compileVectorDimensionCount<T_First>() : 1u>
        {
            static_assert(
                !isCompileVector<T_First> || (compatibleCompileVector<T_First, T_Remaining>() && ...),
                "Every vector in CTypes must have the same number of components.");
        };

        template<typename T_Domain, typename T_Generator, typename... T_Names>
        struct CandidateDimensionCount<domain::Dependent<T_Domain, T_Generator, T_Names...>>
            : CandidateDimensionCount<T_Domain>
        {
        };

        template<typename T_Domain>
        struct CandidateDimensionCount<AutoCandidates<T_Domain>> : CandidateDimensionCount<T_Domain>
        {
        };

        template<typename T>
        struct CandidateDimensionCount<AutoCandidates<RVals<T>>> : RuntimeCandidateDimensionCount<T>
        {
        };

        template<typename T_Values>
        struct CandidateDimensionCount<ListedMdCandidates<T_Values>> : std::integral_constant<std::size_t, 1u>
        {
        };

        template<typename T_Values>
        inline constexpr bool isAutomaticCandidates = requires { T_Values::automaticCandidates; };
        template<typename T_Values>
        inline constexpr std::size_t candidateDimensionCount
            = CandidateDimensionCount<std::remove_cvref_t<T_Values>>::value;

        template<typename T_Values>
        struct IsMdCandidateList : std::false_type
        {
        };

        template<typename T>
        struct IsMdCandidateList<RVals<T>> : std::bool_constant<alpaka::isVector_v<T>>
        {
        };

        template<typename T_First, typename... T_Remaining>
        struct IsMdCandidateList<CTypes<T_First, T_Remaining...>>
            : std::bool_constant<isCompileVector<T_First> && (compatibleCompileVector<T_First, T_Remaining>() && ...)>
        {
        };

        template<typename T_Values, std::size_t T_dimension, std::size_t... T_candidate>
        consteval auto compileComponents(std::index_sequence<T_candidate...>)
        {
            using Tuple = typename T_Values::Values;
            return std::array{compileVectorComponent<std::tuple_element_t<T_candidate, Tuple>, T_dimension>()...};
        }

        template<typename T_Values, std::size_t T_dimension>
        consteval auto compileUniqueComponentCount() -> std::size_t
        {
            constexpr auto components
                = compileComponents<T_Values, T_dimension>(std::make_index_sequence<T_Values::size>{});
            auto count = std::size_t{};
            for(std::size_t candidate = 0u; candidate < components.size(); ++candidate)
            {
                auto duplicate = false;
                for(std::size_t previous = 0u; previous < candidate; ++previous)
                    if(components[previous] == components[candidate])
                        duplicate = true;
                if(!duplicate)
                    ++count;
            }
            return count;
        }

        template<typename T_Values, std::size_t T_dimension, std::size_t T_uniqueIndex>
        consteval auto compileUniqueComponent()
        {
            constexpr auto components
                = compileComponents<T_Values, T_dimension>(std::make_index_sequence<T_Values::size>{});
            auto unique = std::size_t{};
            for(std::size_t candidate = 0u; candidate < components.size(); ++candidate)
            {
                auto duplicate = false;
                for(std::size_t previous = 0u; previous < candidate; ++previous)
                    if(components[previous] == components[candidate])
                        duplicate = true;
                if(!duplicate)
                {
                    if(unique == T_uniqueIndex)
                        return components[candidate];
                    ++unique;
                }
            }
            throw "A compile-time vector component index is out of range.";
        }

        template<typename T_Prototype, auto... T_components>
        struct RebindCompileVector
        {
            static_assert(alpaka::isCVector_v<T_Prototype>);
            using type = alpaka::
                CVec<typename T_Prototype::value_type, static_cast<typename T_Prototype::value_type>(T_components)...>;
        };

        template<typename T, T... T_original, auto... T_components>
        struct RebindCompileVector<std::integer_sequence<T, T_original...>, T_components...>
        {
            using type = std::integer_sequence<T, static_cast<T>(T_components)...>;
        };

        template<typename T_Prototype, auto... T_components>
        using RebindCompileVector_t = typename RebindCompileVector<T_Prototype, T_components...>::type;

        template<typename T_Values, typename T_Prototype, typename T_SelectedIndices>
        struct ReconstructCompileVector;

        template<typename T_Values, typename T_Prototype, std::size_t... T_selected>
        struct ReconstructCompileVector<T_Values, T_Prototype, std::index_sequence<T_selected...>>
        {
            static_assert(sizeof...(T_selected) == candidateDimensionCount<T_Values>);

            template<std::size_t... T_dimension>
            static auto make(std::index_sequence<T_dimension...>)
                -> RebindCompileVector_t<T_Prototype, compileUniqueComponent<T_Values, T_dimension, T_selected>()...>;

            using type = decltype(make(std::make_index_sequence<candidateDimensionCount<T_Values>>{}));
        };

        template<typename T_Values, typename T_Prototype, typename T_SelectedIndices>
        using ReconstructCompileVector_t =
            typename ReconstructCompileVector<T_Values, T_Prototype, T_SelectedIndices>::type;

        template<typename T_Sequence, std::size_t T_value>
        struct AppendIndexSequence;

        template<std::size_t... T_values, std::size_t T_value>
        struct AppendIndexSequence<std::index_sequence<T_values...>, T_value>
        {
            using type = std::index_sequence<T_values..., T_value>;
        };

        template<typename T_Sequence, std::size_t T_value>
        using AppendIndexSequence_t = typename AppendIndexSequence<T_Sequence, T_value>::type;

    } // namespace detail

    template<auto T_maximum>
    using CValsTo = typename detail::MakeCValsRange<0, T_maximum, 1>::type;

    template<auto T_minimum, auto T_maximum, auto T_step = 1>
    using CValsRange = typename detail::MakeCValsRange<T_minimum, T_maximum, T_step>::type;

    template<FixedString T_name, typename T_Values>
    struct Tunable
    {
        static constexpr auto name = T_name;
        using ValuesType = T_Values;
        T_Values values;

        constexpr Tunable(TunableName<T_name>, T_Values candidateValues) : values(std::move(candidateValues))
        {
            static_assert(
                detail::isRVals<T_Values> || detail::isCompileTimeCandidates<T_Values>,
                "A tunable needs RVals, CVals, or CTypes.");
        }

        [[nodiscard]] static constexpr auto nameView() -> std::string_view
        {
            return T_name.view();
        }
    };

    template<FixedString T_name, typename T_Values>
    Tunable(TunableName<T_name>, T_Values) -> Tunable<T_name, T_Values>;

    template<FixedString T_name, typename T_Values>
    using NamedTunable = Tunable<T_name, T_Values>;

    template<FixedString T_name>
    template<typename T_Values>
    [[nodiscard]] constexpr auto TunableName<T_name>::operator()(T_Values values) const
        -> Tunable<T_name, std::remove_cvref_t<T_Values>>
    {
        using CandidateValues = std::remove_cvref_t<T_Values>;
        static_assert(
            detail::isRVals<CandidateValues> || detail::isCompileTimeCandidates<CandidateValues>,
            "A tunable needs RVals, CVals, or CTypes.");
        return {*this, std::move(values)};
    }

    template<FixedString T_name>
    template<typename T_Values>
    [[nodiscard]] constexpr auto TunableName<T_name>::operator()(T_Values values, mdPolicy::Independent) const
    {
        static_assert(
            detail::IsMdCandidateList<T_Values>::value,
            "An mdPolicy requires an explicit RVals vector list or CTypes "
            "vectors with matching dimensions.");
        return (*this)(std::move(values));
    }

    template<FixedString T_name>
    template<typename T_Values>
    [[nodiscard]] constexpr auto TunableName<T_name>::operator()(T_Values values, mdPolicy::Listed) const
    {
        static_assert(
            detail::IsMdCandidateList<T_Values>::value,
            "An mdPolicy requires an explicit RVals vector list or CTypes "
            "vectors with matching dimensions.");
        return (*this)(detail::ListedMdCandidates<T_Values>{std::move(values)});
    }

    struct Named
    {
        template<typename T_Name, typename T_Values>
        [[nodiscard]] constexpr auto operator()(T_Name name, T_Values values) const
        {
            return name(std::move(values));
        }

        template<typename T_Name, typename T_Values, concepts::MdPolicy T_Policy>
        [[nodiscard]] constexpr auto operator()(T_Name name, T_Values values, T_Policy policy) const
        {
            return name(std::move(values), policy);
        }
    };

    inline constexpr Named named{};

    namespace detail
    {

        template<typename T_Component, typename = void>
        struct ComponentEntries
        {
            using type = std::tuple<std::remove_cvref_t<T_Component>>;
        };

        template<typename T_Component>
        struct ComponentEntries<T_Component, std::void_t<typename std::remove_cvref_t<T_Component>::EntriesType>>
        {
            using type = typename std::remove_cvref_t<T_Component>::EntriesType;
        };

        template<typename T_Component>
        using ComponentEntries_t = typename ComponentEntries<T_Component>::type;

        template<typename T_Component, typename = void>
        struct ComponentRestrictions
        {
            using type = std::tuple<>;
        };

        template<typename T_Component>
        struct ComponentRestrictions<
            T_Component,
            std::void_t<typename std::remove_cvref_t<T_Component>::RestrictionsType>>
        {
            using type = typename std::remove_cvref_t<T_Component>::RestrictionsType;
        };

        template<typename T_Component>
        using ComponentRestrictions_t = typename ComponentRestrictions<T_Component>::type;

        template<typename... T_Tuples>
        using TupleCat_t = decltype(std::tuple_cat(std::declval<T_Tuples>()...));

        template<typename T_Component>
        [[nodiscard]] constexpr auto componentEntries(T_Component&& component)
        {
            if constexpr(requires { typename std::remove_cvref_t<T_Component>::EntriesType; })
            {
                return std::apply(
                    [](auto&&... entries)
                    {
                        return std::tuple<std::remove_cvref_t<decltype(entries)>...>{
                            std::forward<decltype(entries)>(entries)...};
                    },
                    std::forward<T_Component>(component).entries());
            }
            else
            {
                using Entry = std::remove_cvref_t<T_Component>;
                static_assert(
                    requires { Entry::name; },
                    "TunableBundle components must be named tunables or tuning "
                    "space fragments.");
                return std::tuple<Entry>{std::forward<T_Component>(component)};
            }
        }

        template<typename T_Component>
        [[nodiscard]] constexpr auto componentRestrictions(T_Component&& component)
        {
            if constexpr(requires { typename std::remove_cvref_t<T_Component>::RestrictionsType; })
            {
                return std::apply(
                    [](auto&&... restrictions)
                    {
                        return std::tuple<std::remove_cvref_t<decltype(restrictions)>...>{
                            std::forward<decltype(restrictions)>(restrictions)...};
                    },
                    std::forward<T_Component>(component).restrictions());
            }
            else
            {
                return std::tuple<>{};
            }
        }

        template<typename T_Tuple>
        struct TupleEntriesAreNamed;

        template<typename T_Entry, typename = void>
        struct EntryIsNamed : std::false_type
        {
        };

        template<typename T_Entry>
        struct EntryIsNamed<T_Entry, std::void_t<decltype(T_Entry::name)>> : std::true_type
        {
        };

        template<typename... T_Entries>
        struct TupleEntriesAreNamed<std::tuple<T_Entries...>>
            : std::bool_constant<(EntryIsNamed<T_Entries>::value && ...)>
        {
        };

    } // namespace detail

    /**
     * Heterogeneous, statically typed bundle of tunable parameters.
     *
     * A component can be one named tunable or a tuning-space fragment.  Fragment
     * entries and restrictions are flattened into the resulting bundle.
     */
    template<typename... T_Components>
    class TunableBundle
    {
    public:
        using EntriesType = detail::TupleCat_t<detail::ComponentEntries_t<T_Components>...>;
        using RestrictionsType = detail::TupleCat_t<detail::ComponentRestrictions_t<T_Components>...>;

        explicit constexpr TunableBundle(T_Components... components)
            : m_entries(std::tuple_cat(detail::componentEntries(std::move(components))...))
            , m_restrictions(std::tuple_cat(detail::componentRestrictions(std::move(components))...))
        {
            static_assert(
                detail::TupleEntriesAreNamed<EntriesType>::value,
                "TunableBundle entries must be bound to tunable names.");
        }

        [[nodiscard]] constexpr auto entries() const& noexcept -> EntriesType const&
        {
            return m_entries;
        }

        [[nodiscard]] constexpr auto entries() && noexcept -> EntriesType&&
        {
            return std::move(m_entries);
        }

        [[nodiscard]] constexpr auto restrictions() const& noexcept -> RestrictionsType const&
        {
            return m_restrictions;
        }

        [[nodiscard]] constexpr auto restrictions() && noexcept -> RestrictionsType&&
        {
            return std::move(m_restrictions);
        }

        static constexpr auto size = std::tuple_size_v<EntriesType>;

    private:
        EntriesType m_entries;
        RestrictionsType m_restrictions;
    };

    template<typename... T_Components>
    TunableBundle(T_Components...) -> TunableBundle<T_Components...>;

    template<typename... T_Components>
    using Tunables = TunableBundle<T_Components...>;

    namespace detail
    {
        template<typename T>
        struct TunablesTraits;

        template<typename T>
        struct IsTunableBundle : std::false_type
        {
        };

        template<typename... T_Components>
        struct IsTunableBundle<TunableBundle<T_Components...>> : std::true_type
        {
        };

        template<typename T>
        inline constexpr bool isTunableBundle = IsTunableBundle<std::remove_cvref_t<T>>::value;
    } // namespace detail

    /**
     * A lazy relationship between the values of two named tuning parameters.
     *
     * The ordinary TunableBundle product remains the default.  This wrapper is
     * useful when two otherwise tunable values must move together, for example a
     * frame extent and the matching number of frames for a fixed problem size.
     */
    template<FixedString T_first, FixedString T_second, typename T_Predicate>
    class Restriction
    {
    public:
        static constexpr auto first = T_first;
        static constexpr auto second = T_second;

        explicit Restriction(T_Predicate predicate) : m_predicate(std::move(predicate))
        {
        }

        template<alpaka::concepts::VectorOrScalar T_First, alpaka::concepts::VectorOrScalar T_Second>
        [[nodiscard]] auto accepts(T_First const& firstValue, T_Second const& secondValue) const -> bool
        {
            return static_cast<bool>(m_predicate(firstValue, secondValue));
        }

    private:
        T_Predicate m_predicate;
    };

    template<typename T_FirstName, typename T_SecondName, typename T_Predicate>
    requires requires {
        std::remove_cvref_t<T_FirstName>::value;
        std::remove_cvref_t<T_SecondName>::value;
    }
    [[nodiscard]] auto restrict(T_FirstName, T_SecondName, T_Predicate predicate)
    {
        constexpr auto first = std::remove_cvref_t<T_FirstName>::value;
        constexpr auto second = std::remove_cvref_t<T_SecondName>::value;
        using Result = Restriction<first, second, std::remove_cvref_t<T_Predicate>>;
        return Result{std::move(predicate)};
    }

    template<FixedString T_parameter, typename T_Predicate>
    class UnaryRestriction
    {
    public:
        static constexpr auto first = T_parameter;

        explicit UnaryRestriction(T_Predicate predicate) : m_predicate(std::move(predicate))
        {
        }

        template<alpaka::concepts::VectorOrScalar T>
        [[nodiscard]] auto accepts(T const& value) const -> bool
        {
            return static_cast<bool>(m_predicate(value));
        }

    private:
        T_Predicate m_predicate;
    };

    template<typename T_Name, typename T_Predicate>
    requires requires { std::remove_cvref_t<T_Name>::value; }
    [[nodiscard]] auto restrict(T_Name, T_Predicate predicate)
    {
        constexpr auto name = std::remove_cvref_t<T_Name>::value;
        using Result = UnaryRestriction<name, std::remove_cvref_t<T_Predicate>>;
        return Result{std::move(predicate)};
    }

    template<typename T_TunablesType, typename... T_Restrictions>
    class ConstrainedTunables
    {
    public:
        using EntriesType = typename T_TunablesType::EntriesType;
        using RestrictionsType = std::tuple<T_Restrictions...>;
        static constexpr auto size = T_TunablesType::size;

        constexpr ConstrainedTunables(T_TunablesType tunables, RestrictionsType restrictions)
            : m_tunables(std::move(tunables))
            , m_restrictions(std::move(restrictions))
        {
        }

        [[nodiscard]] constexpr auto entries() const& noexcept -> decltype(auto)
        {
            return m_tunables.entries();
        }

        [[nodiscard]] constexpr auto entries() && noexcept -> decltype(auto)
        {
            return std::move(m_tunables).entries();
        }

        [[nodiscard]] constexpr auto restrictions() const& noexcept -> RestrictionsType const&
        {
            return m_restrictions;
        }

        [[nodiscard]] constexpr auto restrictions() && noexcept -> RestrictionsType&&
        {
            return std::move(m_restrictions);
        }

    private:
        T_TunablesType m_tunables;
        std::tuple<T_Restrictions...> m_restrictions;
    };

    namespace detail
    {

        template<typename T_TunablesType, typename T_RestrictionsTuple>
        struct ConstrainedTunablesFromTuple;

        template<typename T_TunablesType, typename... T_Restrictions>
        struct ConstrainedTunablesFromTuple<T_TunablesType, std::tuple<T_Restrictions...>>
        {
            using type = ConstrainedTunables<T_TunablesType, T_Restrictions...>;
        };

        template<typename T_TunablesType, typename T_RestrictionsTuple>
        using ConstrainedTunablesFromTuple_t =
            typename ConstrainedTunablesFromTuple<T_TunablesType, T_RestrictionsTuple>::type;

    } // namespace detail

    /** Limit a tuning space with lazy relationships between named parameters. */
    template<typename T_TunablesType, typename... T_Restrictions>
    [[nodiscard]] constexpr auto constrain(T_TunablesType tunables, T_Restrictions... restrictions)
    {
        static_assert((requires {
                          std::remove_cvref_t<T_Restrictions>::first;
                      } && ...), "constrain accepts relations created with alpakaTune::restrict.");
        using Traits = detail::TunablesTraits<T_TunablesType>;
        static_assert(
            ((Traits::template has<std::remove_cvref_t<T_Restrictions>::first> &&
              []
              {
                  if constexpr(requires { std::remove_cvref_t<T_Restrictions>::second; })
                      return Traits::template has<std::remove_cvref_t<T_Restrictions>::second>;
                  else
                      return true;
              }())
             && ...),
            "Both parameters of every restriction must name tunables in this tuning "
            "space.");
        auto allRestrictions = std::tuple_cat(
            detail::componentRestrictions(std::move(tunables)),
            std::tuple<std::remove_cvref_t<T_Restrictions>...>{std::move(restrictions)...});
        using Result = detail::ConstrainedTunablesFromTuple_t<T_TunablesType, decltype(allRestrictions)>;
        return TunableBundle{Result{std::move(tunables), std::move(allRestrictions)}};
    }

    /** Placeholder argument in a prototype alpaka::KernelBundle. */
    template<FixedString T_name>
    struct MarkedTunable
    {
        static constexpr auto name = T_name;
        static constexpr bool tunableMarker = true;
    };

    template<typename T_Name>
    requires requires { std::remove_cvref_t<T_Name>::value; }
    [[nodiscard]] constexpr auto markTunable(T_Name)
    {
        constexpr auto name = std::remove_cvref_t<T_Name>::value;
        return MarkedTunable<name>{};
    }

    namespace detail
    {

        template<typename T>
        inline constexpr bool isMarkedTunable = requires { std::remove_cvref_t<T>::tunableMarker; };

        template<FixedString T_left, FixedString T_right>
        inline constexpr bool sameName = T_left.view() == T_right.view();

        template<FixedString T_name, typename... T_Entries>
        inline constexpr bool hasEntry = (sameName<T_name, T_Entries::name> || ...);

        template<FixedString T_name, typename... T_Entries>
        consteval auto findEntryIndexValue() -> std::size_t
        {
            static_assert(hasEntry<T_name, T_Entries...>, "A KernelBundle marker refers to an unknown tunable.");
            constexpr auto matches = std::array{sameName<T_name, T_Entries::name>...};
            for(auto index = std::size_t{}; index < matches.size(); ++index)
                if(matches[index])
                    return index;
            return matches.size();
        }

        template<FixedString T_name, typename... T_Entries>
        inline constexpr std::size_t findEntryIndex = findEntryIndexValue<T_name, T_Entries...>();

        template<FixedString T_name, typename... T_Entries>
        using FindEntry = std::tuple_element_t<findEntryIndex<T_name, T_Entries...>, std::tuple<T_Entries...>>;

        template<FixedString T_name, typename... T_Entries>
        consteval auto findEntryDimensionOffsetValue() -> std::size_t
        {
            static_assert(hasEntry<T_name, T_Entries...>, "An unknown tunable has no dimension offset.");
            constexpr auto matches = std::array{sameName<T_name, T_Entries::name>...};
            constexpr auto dimensionCounts = std::array{candidateDimensionCount<typename T_Entries::ValuesType>...};
            auto offset = std::size_t{};
            for(auto index = std::size_t{}; index < matches.size(); ++index)
            {
                if(matches[index])
                    return offset;
                offset += dimensionCounts[index];
            }
            return offset;
        }

        template<FixedString T_name, typename... T_Entries>
        inline constexpr std::size_t findEntryDimensionOffset = findEntryDimensionOffsetValue<T_name, T_Entries...>();

        template<FixedString T_name, typename... T_Entries>
        constexpr auto tupleEntry(std::tuple<T_Entries...> const& entries) -> FindEntry<T_name, T_Entries...> const&
        {
            return std::get<findEntryIndex<T_name, T_Entries...>>(entries);
        }

        template<typename T_Tuple>
        struct TunablesTraitsFromTuple;

        template<typename... T_Entries>
        struct TunablesTraitsFromTuple<std::tuple<T_Entries...>>
        {
            using EntriesType = std::tuple<T_Entries...>;
            static constexpr std::size_t size = sizeof...(T_Entries);
            static constexpr std::size_t dimensionCount
                = (candidateDimensionCount<typename T_Entries::ValuesType> + ... + 0u);
            static constexpr bool generatesCandidates
                = (isAutomaticCandidates<typename T_Entries::ValuesType> || ... || false);
            static constexpr bool enumerable = ([] {
    using Values = typename T_Entries::ValuesType;
    if constexpr (requires { Values::enumerable; })
      return Values::enumerable;
    else
      return true;
  }() && ... && true);

            template<FixedString T_name>
            static constexpr bool has = hasEntry<T_name, T_Entries...>;

            template<FixedString T_name>
            using Entry = FindEntry<T_name, T_Entries...>;

            template<FixedString T_name>
            static constexpr std::size_t index = findEntryIndex<T_name, T_Entries...>;

            template<FixedString T_name>
            static constexpr std::size_t dimensionOffset = findEntryDimensionOffset<T_name, T_Entries...>;
        };

        template<typename... T_Components>
        struct TunablesTraits<TunableBundle<T_Components...>>
            : TunablesTraitsFromTuple<typename TunableBundle<T_Components...>::EntriesType>
        {
        };

        template<typename T_TunablesType, typename... T_Restrictions>
        struct TunablesTraits<ConstrainedTunables<T_TunablesType, T_Restrictions...>> : TunablesTraits<T_TunablesType>
        {
        };

    } // namespace detail

} // namespace alpakaTune
