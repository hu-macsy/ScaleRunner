#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include "graph.h"
#include "mockup.h"

#include <scalerunner/first_order_rw.h>
#include <scalerunner/graph.h>
#include <scalerunner/utils.h>

#include <dhb/dynamic_hashed_blocks.h>
#include <gdsb/graph_input.h>

#include <istream>
#include <optional>

TEST_CASE("Simple Random Walk") {
    dhb::Weight constexpr default_weight = 1.f;
    dhb::EdgeID constexpr default_id = 0u;
    dhb::Edges edges;
    auto emplace = [&](dhb::Vertex u, dhb::Vertex v) {
        edges.push_back(dhb::Edge{u, dhb::Target{v, dhb::EdgeData{default_weight, default_id}}});
    };
    std::ifstream bio_celegans_input(bio_celegans_graph_path);
    auto const [vertex_count, edge_count] =
        gdsb::read_graph<dhb::Vertex, decltype(emplace),
                         gdsb::MatrixMarketUndirectedUnweightedNoLoopStatic>(bio_celegans_input,
                                                                             std::move(emplace));

    SECTION("starting from vertex 1") {
        dhb::Matrix<sr::Weight> m(vertex_count);
        for (auto e : edges) {
            m.insert(e.source, e.target.vertex, e.target.data.weight);
        }

        constexpr size_t length = 10;

        std::random_device device;
        auto rng = sr::make_rng_int<size_t>(device);

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            return sr::first_order::step(n, rng);
        };
        sr::Path const path = sr::first_order::walk(std::move(step_f), m, 1, length);
        REQUIRE(path.size() == length);
    }

    SECTION("starting from random vertex") {
        dhb::Matrix<sr::Weight> m(vertex_count);
        for (auto e : edges) {
            m.insert(e.source, e.target.vertex, e.target.data.weight);
        }

        constexpr size_t length = 10;

        std::random_device device;
        auto rng = sr::make_rng_int<size_t>(device);

        auto random_range = Catch::Generators::RandomIntegerGenerator<dhb::Vertex>(
            0u, m.vertices_count() - 1, 1337);
        random_range.next();
        dhb::Vertex const random_vertex = random_range.get();
        // dhb::Vertex const random_vertex = random(0u, m.vertices_count() - 1);

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            return sr::first_order::step(n, rng);
        };

        sr::Path const path = sr::first_order::walk(std::move(step_f), m, random_vertex, length);
        REQUIRE(path.size() == length);
    }

    SECTION("deterministic") {
        dhb::Matrix<sr::Weight> m(vertex_count);
        for (auto e : edges) {
            m.insert(e.source, e.target.vertex, e.target.data.weight);
        }
        constexpr size_t length = 10;
        constexpr dhb::Vertex start_vertex = 1;

        std::vector<dhb::Vertex> series{186, 3, 408, 33, 147, 68, 230, 168, 229};
        sr::Path expected_path;
        for (auto e : series) {
            expected_path.push_back(e);
        }

        std::vector<size_t> indices = construct_indices(m, series, start_vertex);
        SeriesT<size_t> me(std::move(indices));

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            return sr::first_order::step(n, me);
        };

        sr::Path const path = sr::first_order::walk(std::move(step_f), m, 1, length);
        REQUIRE(path.size() == length);

        expected_path.insert(expected_path.begin(), start_vertex);
        REQUIRE(expected_path.size() == length);
        REQUIRE(path[0] == start_vertex);

        CHECK(path == expected_path);
    }
}