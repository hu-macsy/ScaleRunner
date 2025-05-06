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

TEST_CASE("Weighted Random Walk") {
    dhb::EdgeID constexpr default_id = 0u;
    dhb::Edges edges;
    auto emplace = [&](dhb::Vertex const u, dhb::Vertex const v, dhb::Weight const w) {
        edges.push_back(dhb::Edge{u, dhb::Target{v, dhb::EdgeData{w, default_id}}});
    };

    std::ifstream aves_songbird_social_input(aves_songbird_social_graph_path);
    auto const [vertex_count, edge_count] =
        gdsb::read_graph<dhb::Vertex, decltype(emplace),
                         gdsb::EdgeListUndirectedWeightedNoLoopStatic>(aves_songbird_social_input,
                                                                       std::move(emplace));

    SECTION("first test") {
        dhb::Matrix<sr::Weight> m(vertex_count);
        for (auto e : edges) {
            m.insert(e.source, e.target.vertex, e.target.data.weight);
        }

        constexpr size_t length = 10;

        std::random_device device;
        auto f_rng = sr::make_rng_real<float>(device);
        auto i_rng = sr::make_rng_int<size_t>(device);

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            return sr::first_order::step_weighted(n, i_rng, f_rng);
        };

        sr::Path const path = sr::first_order::walk(std::move(step_f), m, 1, length);
        REQUIRE(path.size() == length);
    }
}

TEST_CASE("Weighted Deterministic Walk") {
    dhb::EdgeID constexpr default_id = 0u;
    dhb::Edges edges;
    auto emplace = [&](dhb::Vertex const u, dhb::Vertex const v, dhb::Weight const w) {
        edges.push_back(dhb::Edge{u, dhb::Target{v, dhb::EdgeData{w, default_id}}});
    };

    std::ifstream aves_thornbill_farine_input(aves_thornbill_farine_graph_path);
    auto const [vertex_count, edge_count] =
        gdsb::read_graph<dhb::Vertex, decltype(emplace),
                         gdsb::EdgeListUndirectedWeightedNoLoopStatic>(aves_thornbill_farine_input,
                                                                       std::move(emplace));
    dhb::Matrix<sr::Weight> graph(vertex_count);
    for (auto e : edges) {
        graph.insert(e.source, e.target.vertex, e.target.data.weight);
    }

    SECTION("deterministic") {
        graph.for_nodes([&graph](dhb::Vertex const u) {
            graph.sort(u, [](dhb::BlockState<float>::Entry& a, dhb::BlockState<float>::Entry& b) {
                return a.vertex < b.vertex;
            });
        });

        constexpr dhb::Vertex start_vertex = 1;
        sr::Path expected_path{start_vertex};
        std::vector<dhb::Vertex> determined_path;
        std::vector<sr::Weight> probability_values;

        // As in the 1st order RW test code for deterministic RW algorithms, we
        // want to have each probability be less or equal the weight of the
        // edge. In this case we'll start with a probability that's actually
        // greater than the weight of edge {1, 10}. Therefore, for rejection
        // sampling, this will simply shoot another random point.
        sr::Weight constexpr prob_greater_than_edge_1_to_10 = 6.f + 1.f;
        probability_values.push_back(prob_greater_than_edge_1_to_10);
        // We also need a vertex to draw from first (or actually an index to
        // choose "randomly" from). But we add that later after we resolved the
        // indices: otherwise the index computation is wrong.

        // The next one is actually a valid probability less than 6 as is the
        // weight of edge {1, 10}
        sr::Weight constexpr probability_1_to_10 = 4.f;
        probability_values.push_back(probability_1_to_10);
        dhb::Vertex constexpr v10 = 10u;
        determined_path.push_back(v10);
        expected_path.push_back(v10);

        sr::Weight const probability_10_to_6 = 1.5f;
        probability_values.push_back(probability_10_to_6);
        dhb::Vertex constexpr v6 = 6u;
        determined_path.push_back(v6);
        expected_path.push_back(v6);

        sr::Weight const probability_6_to_2 = .9f;
        probability_values.push_back(probability_6_to_2);
        dhb::Vertex constexpr v2 = 2;
        determined_path.push_back(v2);
        expected_path.push_back(v2);

        // Represents maximum probability for edge {2, 62}
        sr::Weight const probability_2_to_62 = 2.f;
        probability_values.push_back(probability_2_to_62);
        dhb::Vertex constexpr v62 = 62;
        determined_path.push_back(v62);
        expected_path.push_back(v62);

        // The vertex values must be converted to indices within the
        // neighborhood of the graph DS. The graph must be sorted in order to
        // compute the indices correctly.
        std::vector<size_t> indices = construct_indices(graph, determined_path, start_vertex);
        // Now we insert what should be the first index to draw from. In this
        // case it'll be index 0 (we could also use index 99 since the index
        // will never actually be used). Most importantly we add this index
        // after we computed the indices from vertex IDs.
        indices.insert(indices.begin(), 0u);
        SeriesT<size_t> indices_series{std::move(indices)};
        SeriesT<sr::Weight> probability_series{std::move(probability_values)};

        size_t const length = expected_path.size();

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            return sr::first_order::step_weighted(n, indices_series, probability_series);
        };

        sr::Path const path = sr::first_order::walk(std::move(step_f), graph, start_vertex, length);
        REQUIRE(path.size() == length);

        CHECK(start_vertex == path[0]);
        CHECK(expected_path == path);
    }
}