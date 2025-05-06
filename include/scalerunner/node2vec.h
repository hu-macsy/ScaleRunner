#pragma once

#include <scalerunner/first_order_rw.h>
#include <scalerunner/graph.h>
#include <scalerunner/utils.h>

#include <dhb/dynamic_hashed_blocks.h>

#include <cassert>
#include <optional>

namespace sr {

namespace node2vec {
struct Parameter {
    float p_return;
    float q_in_out;
};

struct MaxProbabilityData {
    float max_probability{0.f};
    dhb::Vertex max_probability_vertex{std::numeric_limits<dhb::Vertex>::max()};
    float second_max_probability{0.f};
};

constexpr inline float alpha_out(float const q_in_out) { return 1.f / q_in_out; }

constexpr inline float alpha_return(float const p_return) { return 1.f / p_return; }

constexpr inline float alpha_clique() { return 1.f; }

inline float alpha_prob(dhb::Matrix<sr::Weight> const& graph, dhb::Vertex const next_vertex,
                        dhb::Vertex last_vertex, Parameter const& parameter) {
    if (next_vertex == last_vertex) {
        return sr::node2vec::alpha_return(parameter.p_return);
    }

    if (graph.neighbors(next_vertex).exists(last_vertex)) {
        return sr::node2vec::alpha_clique();
    }

    return sr::node2vec::alpha_out(parameter.q_in_out);
}

constexpr inline float max_alpha(Parameter const& parameter) {
    if (parameter.p_return < 1.f || parameter.q_in_out < 1.f) {
        return std::max(alpha_return(parameter.p_return), alpha_out(parameter.q_in_out));
    }

    return alpha_clique();
}

constexpr inline float second_max_alpha(Parameter const& parameter) {
    float const a_return = alpha_return(parameter.p_return);
    float const a_out = alpha_out(parameter.q_in_out);
    float constexpr a_clique = alpha_clique();
    return second_max(a_return, a_out, a_clique);
}

namespace unweighted {

inline float prob_sum(dhb::Matrix<sr::Weight> const& graph,
                      dhb::Matrix<sr::Weight>::ConstNeighborView v_n, dhb::Vertex last_vertex,
                      Parameter const& parameter) {

    auto acc_f = [&](float acc, dhb::BlockState<float>::const_proxy v) {
        return acc + alpha_prob(graph, v.vertex(), last_vertex, parameter);
    };

    return std::accumulate(std::begin(v_n), std::end(v_n), 0.f, std::move(acc_f));
}

inline float prob_max(dhb::Matrix<sr::Weight> const& graph,
                      dhb::Matrix<sr::Weight>::ConstNeighborView v_n, dhb::Vertex last_vertex,
                      Parameter const& parameter, float const max_absolute) {
    assert(v_n.degree() > 0);

    float max_prob = 0.f;
    for (auto u = std::begin(v_n); u != std::end(v_n); ++u) {
        float const current_prob = alpha_prob(graph, u->vertex(), last_vertex, parameter);
        if (max_absolute == current_prob) {
            return current_prob;
        }

        if (current_prob > max_prob) {
            max_prob = current_prob;
        }
    }

    return max_prob;
}

template <class RandomNumberGenerator, class RandomFloatGenerator>
inline std::optional<dhb::Vertex> step_rejection_sampling(
    dhb::Matrix<sr::Weight> const& graph, dhb::Matrix<sr::Weight>::ConstNeighborView n,
    std::optional<dhb::Vertex> last_vertex, Parameter const& parameter, float const max_alpha,
    RandomNumberGenerator& i_rng, RandomFloatGenerator& f_rng) {
    // First we handle the case that there is no last vertex yet.
    if (!last_vertex) {
        return sr::first_order::step(n, i_rng);
    }
    if (n.degree() == 0) {
        return {};
    }

    if (n.degree() == 1) {
        return n.begin()->vertex();
    }

    float const max_prob = prob_max(graph, n, last_vertex.value(), parameter, max_alpha);
    size_t const max_offset = size_t(n.degree()) - 1;

    while (true) {
        size_t const random_neighbor_offset = i_rng(max_offset);
        auto random_neighbor = std::cbegin(n);
        std::advance(random_neighbor, random_neighbor_offset);

        float const random_prob_shot = f_rng(max_prob);
        float const random_neighbor_prob =
            alpha_prob(graph, random_neighbor->vertex(), last_vertex.value(), parameter);
        if (random_prob_shot <= random_neighbor_prob) {
            return random_neighbor->vertex();
        }
    }
}

} // namespace unweighted

namespace weighted {

inline float prob(dhb::Matrix<sr::Weight> const& graph, dhb::BlockState<float>::const_proxy v,
                  dhb::Vertex last_vertex, Parameter const& parameter) {
    float const alpha = alpha_prob(graph, v.vertex(), last_vertex, parameter);
    return alpha * v.data();
}

inline float prob_sum(dhb::Matrix<sr::Weight> const& graph,
                      dhb::Matrix<sr::Weight>::ConstNeighborView v_n, dhb::Vertex last_vertex,
                      Parameter const& parameter) {

    auto acc_f = [&](float acc, dhb::BlockState<float>::const_proxy v) {
        return acc + prob(graph, v, last_vertex, parameter);
    };

    return std::accumulate(std::begin(v_n), std::end(v_n), 0.f, std::move(acc_f));
}

inline float prob_max(dhb::Matrix<sr::Weight> const& graph,
                      dhb::Matrix<sr::Weight>::ConstNeighborView v_n, dhb::Vertex last_vertex,
                      Parameter const& parameter) {
    assert(v_n.degree() > 0);

    float max_prob = 0.f;
    for (auto u = std::begin(v_n); u != std::end(v_n); ++u) {
        float const current_prob = prob(graph, *u, last_vertex, parameter);
        if (current_prob > max_prob) {
            max_prob = current_prob;
        }
    }

    return max_prob;
}

template <class RandomNumberGenerator, class RandomFloatGenerator>
inline std::optional<dhb::Vertex>
step_rejection_sampling(dhb::Matrix<sr::Weight> const& graph,
                        dhb::Matrix<sr::Weight>::ConstNeighborView n,
                        std::optional<dhb::Vertex> last_vertex, Parameter const& parameter,
                        RandomNumberGenerator& i_rng, RandomFloatGenerator& f_rng) {
    // First we handle the case that there is no last vertex yet.
    if (!last_vertex) {
        return sr::first_order::step_weighted(n, i_rng, f_rng);
    }

    if (n.degree() == 0) {
        return {};
    }

    if (n.degree() == 1) {
        return n.begin()->vertex();
    }

    float const max_prob = prob_max(graph, n, last_vertex.value(), parameter);
    size_t const max_offset = int64_t(n.degree()) - 1;

    while (true) {
        size_t const random_neighbor_offset = i_rng(max_offset);
        auto random_neighbor = std::cbegin(n);
        std::advance(random_neighbor, random_neighbor_offset);

        float const random_prob_shot = f_rng(max_prob);
        float const random_neighbor_prob =
            prob(graph, *random_neighbor, last_vertex.value(), parameter);
        if (random_prob_shot <= random_neighbor_prob) {
            return random_neighbor->vertex();
        }
    }
}

} // namespace weighted

} // namespace node2vec

} // namespace sr