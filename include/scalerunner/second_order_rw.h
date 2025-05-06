#pragma once

#include <scalerunner/random_walk.h>

#include <dhb/dynamic_hashed_blocks.h>

#include <cassert>
#include <limits>
#include <optional>
#include <vector>

namespace sr {

template <typename StepF>
void second_order(StepF&& step_f, dhb::Matrix<sr::Weight> const& graph,
                  dhb::Vertex const start_vertex, Path::iterator begin, size_t const length) {
    dhb::Vertex active_vertex{start_vertex};
    dhb::Matrix<sr::Weight>::ConstNeighborView neighborhood = graph.neighbors(active_vertex);

    Path::iterator path_position = begin;
    std::optional<dhb::Vertex> last_vertex{};

    for (size_t s = 0; s < length; ++s, ++path_position) {
        *path_position = active_vertex;

        std::optional<dhb::Vertex> candidate = step_f(neighborhood, last_vertex);
        if (!candidate) {
            return;
        }

        last_vertex = active_vertex;
        active_vertex = candidate.value();

        neighborhood = graph.neighbors(active_vertex);
    }
}

template <typename StepF>
Path second_order(StepF&& step_f, dhb::Matrix<sr::Weight> const& graph,
                  dhb::Vertex const start_vertex, size_t const length) {
    Path path;
    path.resize(length);

    second_order(std::move(step_f), graph, start_vertex, std::begin(path), length);

    return path;
}

} // namespace sr
