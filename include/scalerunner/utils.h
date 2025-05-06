#pragma once

#include <dhb/graph.h>

#include <scalerunner/random_walk.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

namespace sr {

template <class Engine, class Distribution, typename T> class RandNumberGenerator {
  public:
    RandNumberGenerator(Engine&& engine, Distribution&& distribution)
        : m_engine(std::move(engine)), m_distribution(std::move(distribution)) {}

    //! Use this operator to generate a pseudo random number within interval [0,
    //! max].
    T operator()(T const max) {
        T const rand_value = m_distribution(m_engine);

        if constexpr (std::is_integral<T>()) {
            return rand_value % (max + 1);
        }

        assert(max != std::numeric_limits<float>::infinity());

        return std::fmod(rand_value, max);
    }

  private:
    Engine m_engine;
    Distribution m_distribution;
};

using RandomNumberGeneratorFloat =
    sr::RandNumberGenerator<std::mt19937, std::uniform_real_distribution<float>, float>;
using RandomNumberGeneratorSizeT =
    sr::RandNumberGenerator<std::mt19937, std::uniform_int_distribution<size_t>, size_t>;
using RandomNumberGeneratorVertex =
    sr::RandNumberGenerator<std::mt19937, std::uniform_int_distribution<dhb::Vertex>, dhb::Vertex>;

template <typename RealT>
sr::RandNumberGenerator<std::mt19937, std::uniform_real_distribution<RealT>, RealT>
make_rng_real(std::random_device& device) {
    return sr::RandNumberGenerator<std::mt19937, std::uniform_real_distribution<RealT>, RealT>{
        std::mt19937(device()),
        std::uniform_real_distribution<RealT>(RealT(0.), std::numeric_limits<RealT>::max())};
}

template <typename IntT>
sr::RandNumberGenerator<std::mt19937, std::uniform_int_distribution<IntT>, IntT>
make_rng_int(std::random_device& device) {
    return sr::RandNumberGenerator<std::mt19937, std::uniform_int_distribution<IntT>, IntT>{
        std::mt19937(device()),
        std::uniform_int_distribution<IntT>(IntT(0), std::numeric_limits<IntT>::max())};
}

struct RandomNumberGenPack {
    std::random_device device;
    RandomNumberGeneratorFloat f_rng = make_rng_real<float>(device);
    RandomNumberGeneratorSizeT i_rng = make_rng_int<size_t>(device);
    RandomNumberGeneratorVertex v_rng = make_rng_int<dhb::Vertex>(device);
};

template <typename T> constexpr T second_max(T const a, T const b, T const c) {
    if (a > b) {
        // (a, c), b, (c)
        if (b > c) {
            // a, b, c
            return b;
        }

        // (a, c), b
        return std::min(a, c);
    }

    // (b, c), a (c)
    if (a > c) {
        // b, a, c
        return a;
    }

    // (b, c), a
    return std::min(b, c);
}

template <class OStream> void stream(OStream& ostream, Paths const& paths) {
    for (auto path = std::cbegin(paths); path != std::cend(paths); ++path) {
        for (auto v = std::cbegin(*path); v != std::cend(*path); ++v) {
            ostream << *v << " ";
        }
        ostream << "\n";
    }

    ostream << std::endl;
}

// Was used to check if we should use the inversion instead of the rejection
// counting method to choose the next vertex (especially for 2nd order RWs):
// based on experimental results on the effects of cache locality for sorting
// algorithms (shell sort) it should be beneficial to use inversion couting. So
// far, our experiments have not shown that.
//
// This function will remain here for later use.
constexpr inline bool neighborhood_within_cache(size_t const degree) {
    size_t constexpr cache_line = 64u;
    size_t constexpr size_of_edge = sizeof(dhb::BlockState<sr::Weight>::Entry);
    size_t constexpr max_neighbor_size = 16 * cache_line / size_of_edge;

    return degree <= max_neighbor_size;
}

} // namespace sr