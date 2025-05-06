#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "graph.h"
#include "mockup.h"

#include <scalerunner/graph.h>
#include <scalerunner/node2vec.h>
#include <scalerunner/second_order_rw.h>
#include <scalerunner/utils.h>

#include <dhb/dynamic_hashed_blocks.h>

#include <gdsb/graph_input.h>

TEST_CASE("node2vec, weighted, computing probability for each neighbor") {
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

    dhb::Matrix<sr::Weight> graph_init(vertex_count);
    for (auto e : edges) {
        graph_init.insert(e.source, e.target.vertex, e.target.data.weight);
    }
    dhb::Matrix<sr::Weight> const& graph = graph_init;

    SECTION("prob()") {
        dhb::Vertex constexpr last_vertex = 16u;
        dhb::Vertex constexpr current_vertex = 9u;
        dhb::Vertex constexpr next_vertex = 176u;
        constexpr sr::node2vec::Parameter parameter{.5f, 2.f};

        dhb::BlockState<float>::const_proxy next_vertex_proxy =
            *graph.neighbors(current_vertex).iterator_to(next_vertex);

        float const probability =
            sr::node2vec::weighted::prob(graph, next_vertex_proxy, last_vertex, parameter);

        float const applied_alpha =
            sr::node2vec::alpha_prob(graph, next_vertex, last_vertex, parameter);
        float const applied_weight =
            graph.neighbors(current_vertex).iterator_to(next_vertex)->data();
        float const expected_probability = applied_alpha * applied_weight;

        CHECK(expected_probability == Catch::Approx(probability));
    }

    SECTION("prob_sum()") {
        dhb::Vertex constexpr last_vertex = 16u;
        dhb::Vertex constexpr current_vertex = 9u;
        dhb::Vertex constexpr next_vertex = 176u;

        constexpr sr::node2vec::Parameter parameter{.5f, 2.f};

        dhb::BlockState<float>::const_proxy next_vertex_proxy =
            *(static_cast<dhb::Matrix<sr::Weight> const&>(graph)
                  .neighbors(current_vertex)
                  .iterator_to(next_vertex));

        float const result_prob_sum = sr::node2vec::weighted::prob_sum(
            graph, graph.neighbors(current_vertex), last_vertex, parameter);

        float expected_prob_sum = 0.f;
        for (auto neighbor : graph.neighbors(current_vertex)) {
            float a = sr::node2vec::alpha_prob(graph, neighbor.vertex(), last_vertex, parameter);
            float w = neighbor.data();
            expected_prob_sum += a * w;
        }

        CHECK(expected_prob_sum == Catch::Approx(result_prob_sum));
    }

    SECTION("prob_max()") {
        dhb::Vertex constexpr last_vertex = 16u;
        dhb::Vertex constexpr current_vertex = 9u;
        dhb::Vertex constexpr next_vertex = 176u;

        constexpr sr::node2vec::Parameter parameter{.5f, 2.f};

        dhb::BlockState<float>::const_proxy next_vertex_proxy =
            *(static_cast<dhb::Matrix<sr::Weight> const&>(graph)
                  .neighbors(current_vertex)
                  .iterator_to(next_vertex));

        float const result_prob_max = sr::node2vec::weighted::prob_max(
            graph, graph.neighbors(current_vertex), last_vertex, parameter);

        float expected_prob_max = 0.f;
        for (auto neighbor : graph.neighbors(current_vertex)) {
            float const a =
                sr::node2vec::alpha_prob(graph, neighbor.vertex(), last_vertex, parameter);
            float const w = neighbor.data();

            float const f = a * w;
            if (f > expected_prob_max) {
                expected_prob_max = f;
            }
        }

        CHECK(expected_prob_max == Catch::Approx(result_prob_max));
    }
}

TEST_CASE("node2vec, weighted, make a step") {
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

    dhb::Matrix<sr::Weight> graph_init(vertex_count);
    for (auto e : edges) {
        graph_init.insert(e.source, e.target.vertex, e.target.data.weight);
    }
    dhb::Matrix<sr::Weight> const& graph = graph_init;

    SECTION("first step, no last vertex") {
        std::optional<dhb::Vertex> last_vertex{};
        dhb::Vertex constexpr current_vertex = 1u;

        constexpr sr::node2vec::Parameter parameter{.5f, 2.f};

        SeriesT<size_t> int_series{{0}};

        // We don't really use the float series but we have to provide one for now.
        SeriesT<float> float_series{{0.3f}};

        dhb::Matrix<sr::Weight>::ConstNeighborView neighbors = graph.neighbors(current_vertex);
        REQUIRE(neighbors.degree() > 0);

        auto chosen_vertex = sr::node2vec::weighted::step_rejection_sampling(
            graph, neighbors, std::optional(last_vertex), parameter, int_series, float_series);

        dhb::Vertex expected_vertex = neighbors.begin()->vertex();

        REQUIRE(chosen_vertex.has_value());
        CHECK(expected_vertex == chosen_vertex.value());
    }

    SECTION("step_weighted_rejection_sampling()") {
        dhb::Vertex constexpr last_vertex = 16u;
        dhb::Vertex constexpr current_vertex = 9u;
        dhb::Vertex constexpr next_vertex = 176u;

        std::optional<dhb::Vertex> last_vertex_o{last_vertex};
        constexpr sr::node2vec::Parameter parameter{.5f, 2.f};

        dhb::Matrix<sr::Weight>::ConstNeighborView neighbors = graph.neighbors(current_vertex);
        REQUIRE(neighbors.degree() > 0);
        size_t constexpr int_series_result = 0u;

        SeriesT<size_t> int_series{{int_series_result}};

        auto expected_neighbor = neighbors.begin();
        float const expected_neighbor_prob =
            sr::node2vec::weighted::prob(graph, *expected_neighbor, last_vertex, parameter);

        SeriesT<float> float_series{{expected_neighbor_prob}};

        auto chosen_vertex = sr::node2vec::weighted::step_rejection_sampling(
            graph, neighbors, last_vertex_o, parameter, int_series, float_series);

        REQUIRE(chosen_vertex.has_value());
        CHECK(expected_neighbor->vertex() == chosen_vertex.value());
    }
}

TEST_CASE("node2vec, weighted, walking") {
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
    dhb::Matrix<sr::Weight> graph_init(vertex_count);
    for (auto e : edges) {
        graph_init.insert(e.source, e.target.vertex, e.target.data.weight);
    }
    dhb::Matrix<sr::Weight> const& graph = graph_init;

    SECTION("10 to 33 steps, 100 times, rejection sampling") {
        constexpr size_t start_vertex = 1u;
        constexpr size_t path_length_min = 10u;
        constexpr size_t path_length_max = 33u;

        constexpr sr::node2vec::Parameter parameter{1.f, 0.5f};

        std::random_device device;
        auto f_rng = sr::make_rng_real<float>(device);
        auto i_rng = sr::make_rng_int<size_t>(device);

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n,
                          std::optional<dhb::Vertex> last_vertex) -> std::optional<dhb::Vertex> {
            auto chosen_vertex = sr::node2vec::weighted::step_rejection_sampling(
                graph, n, last_vertex, parameter, i_rng, f_rng);
            return chosen_vertex;
        };

        uint32_t constexpr repetitions = 100u;
        bool path_size_correct = true;
        size_t length = 10u;
        for (uint32_t c = 0; c < repetitions && path_size_correct; ++c) {
            sr::Path const path = sr::second_order(std::move(step_f), graph, start_vertex, length);
            path_size_correct = path_size_correct && (path.size() == length);
            length = f_rng(path_length_max);
            length = std::max(path_length_min, length);
        }

        CHECK(path_size_correct);
    }
}
