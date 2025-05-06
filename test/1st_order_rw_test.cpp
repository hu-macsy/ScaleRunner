#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include "graph.h"
#include "mockup.h"

#include <scalerunner/first_order_rw.h>
#include <scalerunner/utils.h>

#include <dhb/dynamic_hashed_blocks.h>
#include <gdsb/graph_input.h>

#include <istream>
#include <optional>

TEST_CASE("1st Order RW") {
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

    SECTION("RW with Restart") {
        dhb::Matrix<sr::Weight> graph(vertex_count);
        for (auto e : edges) {
            graph.insert(e.source, e.target.vertex, e.target.data.weight);
        }

        constexpr size_t length = 10u;
        constexpr size_t start_vertex = 1u;
        constexpr float restart_probability = 0.1f;

        std::random_device device;
        auto f_rng = sr::make_rng_real<float>(device);
        auto i_rng = sr::make_rng_int<size_t>(device);

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            auto rwr_condition =
                [&](dhb::Matrix<sr::Weight> const& graph, dhb::Vertex,
                    dhb::Matrix<sr::Weight>::ConstNeighborView) -> std::optional<dhb::Vertex> {
                float const prob = f_rng(1.f);
                if (prob <= restart_probability) {
                    return start_vertex;
                }

                return {};
            };

            auto candidate = rwr_condition(graph, n.source(), n);
            return candidate ? candidate : sr::first_order::step(n, i_rng);
        };
        sr::Path path(length);
        sr::first_order::walk(std::move(step_f), graph, start_vertex, std::begin(path), length);
        REQUIRE(path.size() == length);
    }

    SECTION("deterministic") {
        dhb::Matrix<sr::Weight> graph(vertex_count);
        for (auto e : edges) {
            graph.insert(e.source, e.target.vertex, e.target.data.weight);
        }

        constexpr size_t length = 10;
        constexpr dhb::Vertex start_vertex = 1;

        // The series includes the start vertex in order to find the index to
        // move to in the next iteration.
        std::vector<dhb::Vertex> const series{186, 1, 1, 186, 3, 408, 33, 1, 186};
        std::vector<size_t> indices = construct_indices(graph, series, start_vertex);

        // It's important to note that an index from 1 -> 1 can not be found as
        // well as for 33 -> 1. Therefore, the index size is 7.
        REQUIRE(indices.size() == 7);

        // Now we must also remove the element at position 1, since that one is
        // just the just back to the start vertex that shall not be represented
        // in the series that the mockup engine produces.
        indices.erase(std::next(std::begin(indices)));
        REQUIRE(indices.size() == 6);

        SeriesT<size_t> me(std::move(indices));

        SeriesT<bool> condition_series{
            std::vector<bool>{false, true, true, false, false, false, false, true, false}};

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            auto rwr_condition =
                [&](dhb::Matrix<sr::Weight> const&, dhb::Vertex,
                    dhb::Matrix<sr::Weight>::ConstNeighborView) -> std::optional<dhb::Vertex> {
                if (condition_series()) {
                    return start_vertex;
                }

                return {};
            };

            auto candidate = rwr_condition(graph, n.source(), n);
            return candidate ? candidate : sr::first_order::step(n, me);
        };

        sr::Path path(length);
        sr::first_order::walk(std::move(step_f), graph, start_vertex, std::begin(path), length);

        sr::Path expected_path = {start_vertex, 186, 1, 1, 186, 3, 408, 33, 1, 186};

        REQUIRE(path.size() == length);
        CHECK(path == expected_path);
    }
}

TEST_CASE("Page Rank RW") {
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

    SECTION("Using Random Engines") {
        constexpr size_t length = 10u;
        constexpr size_t start_vertex = 1u;
        constexpr float travel_to_random_vertex_p = 0.1f;

        sr::RandomNumberGenPack rngs;

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            auto rwr_condition =
                [&](dhb::Matrix<sr::Weight> const& graph, dhb::Vertex,
                    dhb::Matrix<sr::Weight>::ConstNeighborView) -> std::optional<dhb::Vertex> {
                float const prob = rngs.f_rng(1.f);
                if (prob <= travel_to_random_vertex_p) {
                    return rngs.v_rng(dhb::Vertex(graph.vertices_count() - 1));
                }

                return {};
            };

            auto candidate = rwr_condition(graph, n.source(), n);
            return candidate ? candidate
                             : sr::first_order::step_weighted(n, rngs.i_rng, rngs.f_rng);
        };

        sr::Path const path = sr::first_order::walk(std::move(step_f), graph, start_vertex, length);
        REQUIRE(path.size() != 0u);
    }

    SECTION("Deterministic") {
        graph.for_nodes([&graph](dhb::Vertex const u) {
            graph.sort(u, [](dhb::BlockState<float>::Entry& a, dhb::BlockState<float>::Entry& b) {
                return a.vertex < b.vertex;
            });
        });

        dhb::Vertex constexpr start_vertex = 1u;
        size_t constexpr path_length = 5u;

        // We nee da couple of vectors to determine which values shall be the
        // next "randomly drawn" ones.

        // This will be the weights that will be drawn in sequence. Therefore,
        // if we want to walk along an edge, the weight must be <= the weight of
        // the edge.
        std::vector<float> fp_weight_choice_data;

        // These values determine if we want to terminate the walk for each step.
        std::vector<bool> precondition_values;

        // Will determine which vertex will be chosen next. Thus this will be
        // vertex v if we want to walk along edge {u, v}.
        std::vector<dhb::Vertex> vertex_values;

        sr::Path expected_path{start_vertex};

        // First we want to walk along edge {1, 10}.
        precondition_values.emplace_back(false);
        float constexpr p_v1_to_v10 = 5.f;
        fp_weight_choice_data.emplace_back(p_v1_to_v10);
        dhb::Vertex constexpr v10 = 10u;
        vertex_values.push_back(v10);
        expected_path.push_back(v10);

        // Then we want to walk along edge {10, 6}
        precondition_values.emplace_back(false);
        float const p_v10_to_v6 = 2.f; // represents maximum
        dhb::Vertex constexpr v6 = 6u;
        fp_weight_choice_data.emplace_back(p_v10_to_v6);
        vertex_values.push_back(v6);
        expected_path.push_back(v6);

        // Now we want have the pre-condition to be true and just beam to vertex
        // 38.
        precondition_values.emplace_back(true);
        dhb::Vertex constexpr v38_beam = 38u;
        expected_path.push_back(v38_beam);

        // Finally, we walk along edge {38, 5}
        precondition_values.emplace_back(false);
        float const p_v38_to_v5 = 0.8f;
        dhb::Vertex constexpr v5 = 5u;
        fp_weight_choice_data.emplace_back(p_v38_to_v5);
        vertex_values.push_back(v5);
        expected_path.push_back(v5);

        SeriesT<bool> precondition_series{std::move(precondition_values)};
        SeriesT<float> fp_weight_series{std::move(fp_weight_choice_data)};

        // The vertex values must be converted to indices within the
        // neighborhood of the graph DS. The graph must be sorted in order to
        // compute the indices correctly.
        std::vector<size_t> indices = construct_indices(graph, vertex_values, start_vertex);
        SeriesT<size_t> indices_series{std::move(indices)};

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            auto rwr_condition =
                [&](dhb::Matrix<sr::Weight> const&, dhb::Vertex,
                    dhb::Matrix<sr::Weight>::ConstNeighborView) -> std::optional<dhb::Vertex> {
                if (precondition_series()) {
                    return v38_beam;
                }

                return {};
            };

            auto candidate = rwr_condition(graph, n.source(), n);
            return candidate ? candidate
                             : sr::first_order::step_weighted(n, indices_series, fp_weight_series);
        };

        sr::Path const path =
            sr::first_order::walk(std::move(step_f), graph, start_vertex, path_length);

        REQUIRE(path.size() == path_length);
        CHECK(path == expected_path);
    }
}