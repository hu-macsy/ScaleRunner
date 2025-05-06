#pragma once

#include <scalerunner/graph.h>
#include <scalerunner/random_walk.h>

#include <dhb/dynamic_hashed_blocks.h>

#include <cassert>
#include <limits>
#include <optional>
#include <vector>

namespace sr {

namespace first_order {

template <class RandomNumberGenerator>
std::optional<dhb::Vertex> step(dhb::Matrix<sr::Weight>::ConstNeighborView n,
                                RandomNumberGenerator& rng) {
    if (n.degree() == 0) {
        return {};
    }

    if (n.degree() == 1) {
        return n.begin()->vertex();
    }

    size_t const max_index{n.degree() - 1};
    auto const random_neighbor_index = rng(max_index);
    auto neighbor = n.begin();
    std::advance(neighbor, random_neighbor_index);

    if (neighbor == n.end()) {
        return {};
    }

    return neighbor->vertex();
}

template <class RandomNumberGenerator, class RandomFloatGenerator>
std::optional<dhb::Vertex> step_weighted(dhb::Matrix<sr::Weight>::ConstNeighborView n,
                                         RandomNumberGenerator& i_rng,
                                         RandomFloatGenerator& f_rng) {
    if (n.degree() == 0) {
        return {};
    }

    if (n.degree() == 1) {
        return n.begin()->vertex();
    }

    // We're using rejection sampling for choosing the next edge based on the
    // weight of the edge.
    dhb::BlockState<Weight>::const_iterator const max_weight_edge = std::max_element(
        n.begin(), n.end(),
        [](dhb::BlockState<Weight>::const_proxy a, dhb::BlockState<Weight>::const_proxy b) {
            return a.data() < b.data();
        });

    size_t const max_offset = size_t(n.degree() - 1);

    while (true) {
        size_t const random_neighbor_offset = i_rng(max_offset);
        auto random_neighbor = std::cbegin(n);
        std::advance(random_neighbor, random_neighbor_offset);

        // TODO: The range 0.f to max weight is limits the step to such weight
        // ranges. For graphs with negative weights such behavior will never
        // walk along an edge with negative weight. Moreover, if all edge
        // weights are negative, the result of calling step_weighted() is
        // undefined.
        Weight const random_weight_shot = f_rng(max_weight_edge->data());
        if (random_weight_shot <= random_neighbor->data()) {
            return random_neighbor->vertex();
        }
    }
}

template <typename StepF>
void walk(StepF&& step_f, dhb::Matrix<sr::Weight> const& graph, dhb::Vertex const start_vertex,
          Path::iterator begin, size_t const length) {
    dhb::Vertex active_vertex{start_vertex};
    dhb::Matrix<sr::Weight>::ConstNeighborView neighborhood = graph.neighbors(active_vertex);
    Path::iterator path_position = begin;

    for (size_t s = 0; s < length; ++s, ++path_position) {
        *path_position = active_vertex;

        std::optional<dhb::Vertex> candidate = step_f(neighborhood);
        if (!candidate) {
            return;
        }

        active_vertex = candidate.value();
        neighborhood = graph.neighbors(active_vertex);
    }
}

template <typename StepF>
Path walk(StepF&& step_f, dhb::Matrix<sr::Weight> const& graph, dhb::Vertex const start_vertex,
          size_t const length) {
    Path path(length);
    walk(std::move(step_f), graph, start_vertex, std::begin(path), length);
    return path;
}

} // namespace first_order

} // namespace sr
