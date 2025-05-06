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

TEST_CASE("node2vec, alpha probabilities") {
    SECTION("alpha_out()") {
        float constexpr q = 2.f;
        float constexpr result = sr::node2vec::alpha_out(q);
        CHECK(1.f / q == Catch::Approx(result));
    }

    SECTION("alpha_return()") {
        float constexpr p = 0.5f;
        float constexpr result = sr::node2vec::alpha_return(p);
        CHECK(1.f / p == Catch::Approx(result));
    }

    SECTION("alpha_clique()") {
        float constexpr result = sr::node2vec::alpha_clique();
        CHECK(1.f == Catch::Approx(result));
    }
}

TEST_CASE("node2vec, alpha_prob(), deducing alpha probability") {
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

    SECTION("return edge") {
        dhb::Vertex constexpr last_vertex = 12u;
        dhb::Vertex constexpr current_vertex = 9u;
        dhb::Vertex constexpr next_vertex = 12u;
        constexpr sr::node2vec::Parameter parameter{.5f, 2.f};

        float const result = sr::node2vec::alpha_prob(graph, next_vertex, last_vertex, parameter);

        CHECK(result == sr::node2vec::alpha_return(parameter.p_return));
    }

    SECTION("out edge") {
        dhb::Vertex constexpr last_vertex = 16u;
        dhb::Vertex constexpr current_vertex = 9u;
        dhb::Vertex constexpr next_vertex = 12u;
        constexpr sr::node2vec::Parameter parameter{.5f, 2.f};
        float const result = sr::node2vec::alpha_prob(graph, next_vertex, last_vertex, parameter);

        CHECK(result == sr::node2vec::alpha_out(parameter.q_in_out));
    }

    SECTION("clique edge") {
        dhb::Vertex constexpr last_vertex = 16u;
        dhb::Vertex constexpr current_vertex = 9u;
        dhb::Vertex constexpr next_vertex = 176u;
        constexpr sr::node2vec::Parameter parameter{.5f, 2.f};

        float const result = sr::node2vec::alpha_prob(graph, next_vertex, last_vertex, parameter);

        CHECK(result == sr::node2vec::alpha_clique());
    }
}

TEST_CASE("node2vec, max_alpha()") {
    SECTION("clique probability is max") {
        float constexpr p = 1.5f;
        float constexpr q = 1.8f;
        sr::node2vec::Parameter const parameter{p, q};

        CHECK(sr::node2vec::max_alpha(parameter) == Catch::Approx(sr::node2vec::alpha_clique()));
    }

    SECTION("alpha_return() probability is max") {
        float constexpr p = .5f;
        float constexpr q = .8f;
        sr::node2vec::Parameter const parameter{p, q};

        CHECK(sr::node2vec::max_alpha(parameter) == Catch::Approx(sr::node2vec::alpha_return(p)));
    }

    SECTION("alpha_out() probability is max") {
        float constexpr p = .5f;
        float constexpr q = .3f;
        sr::node2vec::Parameter const parameter{p, q};

        CHECK(sr::node2vec::max_alpha(parameter) == Catch::Approx(sr::node2vec::alpha_out(q)));
    }
}

TEST_CASE("node2vec, second_max_alpha()") {
    SECTION("clique probability is second max") {
        float constexpr p = .5f;
        float constexpr q = 1.5f;
        sr::node2vec::Parameter const parameter{p, q};

        CHECK(sr::node2vec::second_max_alpha(parameter) ==
              Catch::Approx(sr::node2vec::alpha_clique()));
    }

    SECTION("alpha_return() probability is second max") {
        float constexpr p = .8f;
        float constexpr q = .5f;
        sr::node2vec::Parameter const parameter{p, q};

        CHECK(sr::node2vec::second_max_alpha(parameter) ==
              Catch::Approx(sr::node2vec::alpha_return(p)));
    }

    SECTION("alpha_out() probability is second max") {
        float constexpr p = .3f;
        float constexpr q = .5f;
        sr::node2vec::Parameter const parameter{p, q};

        CHECK(sr::node2vec::second_max_alpha(parameter) ==
              Catch::Approx(sr::node2vec::alpha_out(q)));
    }
}

TEST_CASE("node2vec, unweighted, prob_sum() on hu-webgraph") {
    dhb::Weight constexpr default_weight = 1.f;
    dhb::EdgeID constexpr default_id = 0u;
    dhb::Edges edges;
    auto emplace = [&](dhb::Vertex u, dhb::Vertex v) {
        edges.push_back(dhb::Edge{u, dhb::Target{v, dhb::EdgeData{default_weight, default_id}}});
    };
    std::ifstream bio_celegans_input(bio_celegans_graph_path);
    auto const [vertex_count, edge_count] =
        gdsb::read_graph<dhb::Vertex, decltype(emplace),
                         gdsb::EdgeListDirectedUnweightedLoopStatic>(hu_graph_path,
                                                                     std::move(emplace));
    dhb::Matrix<sr::Weight> graph_init(vertex_count);
    for (auto e : edges) {
        graph_init.insert(e.source, e.target.vertex, e.target.data.weight);
    }
    dhb::Matrix<sr::Weight> const& hu_webgraph = graph_init;

    SECTION("probability sum of vertex 10, last vertex 1, p = 1, q = 0.5") {
        sr::node2vec::Parameter parameter_structure{1.f, .5f};

        dhb::Vertex constexpr vertex_10{10u};
        dhb::Vertex constexpr last_vertex_1{1u};
        float const p_sum = sr::node2vec::unweighted::prob_sum(
            hu_webgraph, hu_webgraph.neighbors(vertex_10), last_vertex_1, parameter_structure);
        CHECK(4.f == Catch::Approx(p_sum));
    }

    SECTION("probability sum of vertex 10, last vertex 1, p = .5, q = 2") {
        sr::node2vec::Parameter parameter{.5f, 2.f};

        dhb::Vertex constexpr vertex_10{10u};
        dhb::Vertex constexpr last_vertex_1{1u};
        float const p_sum = sr::node2vec::unweighted::prob_sum(
            hu_webgraph, hu_webgraph.neighbors(vertex_10), last_vertex_1, parameter);
        CHECK(3.5f == Catch::Approx(p_sum));
    }
}

TEST_CASE("node2vec, unweighted, prob_max() on hu-webgraph") {
    dhb::Weight constexpr default_weight = 1.f;
    dhb::EdgeID constexpr default_id = 0u;
    dhb::Edges edges;
    auto emplace = [&](dhb::Vertex u, dhb::Vertex v) {
        edges.push_back(dhb::Edge{u, dhb::Target{v, dhb::EdgeData{default_weight, default_id}}});
    };
    std::ifstream bio_celegans_input(bio_celegans_graph_path);
    auto const [vertex_count, edge_count] =
        gdsb::read_graph<dhb::Vertex, decltype(emplace),
                         gdsb::EdgeListDirectedUnweightedLoopStatic>(hu_graph_path,
                                                                     std::move(emplace));
    dhb::Matrix<sr::Weight> graph_init(vertex_count);
    for (auto e : edges) {
        graph_init.insert(e.source, e.target.vertex, e.target.data.weight);
    }
    dhb::Matrix<sr::Weight> const& hu_webgraph = graph_init;

    SECTION("probability max of vertex 10, last vertex 1, p = 1, q = 0.5") {
        sr::node2vec::Parameter parameter_structure{1.f, .5f};

        dhb::Vertex constexpr vertex_10{10u};
        dhb::Vertex constexpr last_vertex_1{1u};
        float const p_max = sr::node2vec::unweighted::prob_max(
            hu_webgraph, hu_webgraph.neighbors(vertex_10), last_vertex_1, parameter_structure,
            sr::node2vec::max_alpha(parameter_structure));
        float constexpr expected_probabilty_max = 2.f;
        CHECK(expected_probabilty_max == Catch::Approx(p_max));
    }

    SECTION("probability max of vertex 10, last vertex 1, p = .25, q = 2") {
        sr::node2vec::Parameter parameter{.25f, 2.f};

        dhb::Vertex constexpr vertex_10{10u};
        dhb::Vertex constexpr last_vertex_1{1u};
        float const p_max = sr::node2vec::unweighted::prob_max(
            hu_webgraph, hu_webgraph.neighbors(vertex_10), last_vertex_1, parameter,
            sr::node2vec::max_alpha(parameter));
        float constexpr expected_probabilty_max = 4.f;
        CHECK(expected_probabilty_max == Catch::Approx(p_max));
    }
}

TEST_CASE("node2vec, unweighted, make a step") {
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

        auto chosen_vertex = sr::node2vec::unweighted::step_rejection_sampling(
            graph, neighbors, last_vertex, parameter, sr::node2vec::max_alpha(parameter),
            int_series, float_series);

        dhb::Vertex expected_vertex = neighbors.begin()->vertex();

        REQUIRE(chosen_vertex.has_value());
        CHECK(expected_vertex == chosen_vertex.value());
    }

    SECTION("step_rejection_sampling()") {
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
            sr::node2vec::alpha_prob(graph, expected_neighbor->vertex(), last_vertex, parameter);

        SeriesT<float> float_series{{expected_neighbor_prob}};

        auto chosen_vertex = sr::node2vec::unweighted::step_rejection_sampling(
            graph, neighbors, last_vertex_o, parameter, sr::node2vec::max_alpha(parameter),
            int_series, float_series);

        REQUIRE(chosen_vertex.has_value());
        CHECK(expected_neighbor->vertex() == chosen_vertex.value());
    }
}
