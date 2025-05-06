#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include "graph.h"
#include "mockup.h"

#include <scalerunner/graph.h>
#include <scalerunner/rw_scheduler.h>
#include <scalerunner/utils.h>

#include <dhb/dynamic_hashed_blocks.h>
#include <gdsb/graph_input.h>

#include <istream>
#include <optional>
#include <sstream>

TEST_CASE("stream") {
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

    SECTION("to stringstream") {
        dhb::Matrix<sr::Weight> m(vertex_count);
        for (auto e : edges) {
            m.insert(e.source, e.target.vertex, e.target.data.weight);
        }
        constexpr size_t length = 10;
        constexpr dhb::Vertex start_vertex = 1;

        std::vector<dhb::Vertex> series{186, 3, 408, 33, 147, 68, 230, 168, 229};

        auto rw_algorithm = [&]() {
            std::vector<size_t> indices = construct_indices(m, series, start_vertex);
            SeriesT<size_t> me(std::move(indices));
            auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
                return sr::first_order::step(n, me);
            };
            return sr::first_order::walk(std::move(step_f), m, start_vertex, length);
        };

        constexpr size_t rw_count = 3;
        sr::Paths const paths = sr::schedule(std::move(rw_algorithm), rw_count);

        REQUIRE(paths.size() == rw_count);

        sr::Path const& first_path = paths[0];
        sr::Path const& last_path = paths[rw_count - 1];

        REQUIRE(first_path.size() == length);
        REQUIRE(last_path.size() == length);
        REQUIRE(first_path == last_path);

        std::stringstream ss;
        sr::stream(ss, paths);

        std::string const result = ss.str();

        // clang-format off
        std::string const expected_result{
            "1 186 3 408 33 147 68 230 168 229 \n"
            "1 186 3 408 33 147 68 230 168 229 \n"
            "1 186 3 408 33 147 68 230 168 229 \n"
            "\n"};
        // clang-format on

        CHECK(result == expected_result);
    }
}

TEST_CASE("second max") {
    SECTION("int") {
        constexpr int a = 1;
        constexpr int b = 2;
        constexpr int c = 3;

        CHECK(sr::second_max(a, b, c) == b);
    }

    SECTION("uint32_t") {
        constexpr uint32_t a = 4;
        constexpr uint32_t b = 2;
        constexpr uint32_t c = 3;

        CHECK(sr::second_max(a, b, c) == c);
    }

    SECTION("float") {
        constexpr float a = 1.3f;
        constexpr float b = 1.4f;
        constexpr float c = 1.0f;

        CHECK(sr::second_max(a, b, c) == Catch::Approx(a));
    }

    SECTION("double") {
        constexpr double a = 1.8;
        constexpr double b = 1.81;
        constexpr double c = 1.83;

        CHECK(sr::second_max(a, b, c) == Catch::Approx(b));
    }
}

TEST_CASE("neighborhood_within_cache()") {
    SECTION("Fits Within Cache") {
        dhb::Vertex degree = 4u;
        CHECK(sr::neighborhood_within_cache(degree));

        degree = 8u;
        CHECK(sr::neighborhood_within_cache(degree));

        degree = 16u;
        CHECK(sr::neighborhood_within_cache(degree));

        degree = 32u;
        CHECK(sr::neighborhood_within_cache(degree));

        degree = 64u;
        CHECK(sr::neighborhood_within_cache(degree));

        degree = 128u;
        CHECK(sr::neighborhood_within_cache(degree));
    }

    SECTION("Does Not Fit In Cache") {
        dhb::Vertex degree = 256u;
        CHECK(!sr::neighborhood_within_cache(degree));
    }
}