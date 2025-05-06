#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include "graph.h"

#include <scalerunner/graph.h>

#include <dhb/dynamic_hashed_blocks.h>
#include <gdsb/graph_input.h>

#include <istream>

TEST_CASE("Graph Reading, Bio Celegans") {
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

    // we start at 0 so we must have 453 + 1 vertices and not 453 as described on
    // https://networkrepository.com/bio-celegans.php
    unsigned int constexpr bio_celegans_vertex_count = 453 + 1;
    unsigned int constexpr bio_celegans_edge_count = 2025 * 2;
    dhb::Degree constexpr bio_celegans_max_degree = 237;
    std::vector<dhb::Degree> extracted_degrees = dhb::degrees_from(edges);

    SECTION("read data") {
        CHECK(edges.size() == bio_celegans_edge_count);
        CHECK(dhb::graph::vertex_count(edges) == bio_celegans_vertex_count);
        dhb::Degree const max_degree =
            *std::max_element(std::begin(extracted_degrees), std::end(extracted_degrees));

        CHECK(max_degree == bio_celegans_max_degree);
    }

    // TODO: const correctness
    SECTION("initializing DHB matrix") {
        dhb::Matrix<sr::Weight> m(vertex_count);
        for (auto e : edges) {
            m.insert(e.source, e.target.vertex, e.target.data.weight);
        }
        CHECK(m.vertices_count() == bio_celegans_vertex_count);

        size_t const max_degree_vertex = std::distance(
            std::begin(extracted_degrees),
            std::max_element(std::begin(extracted_degrees), std::end(extracted_degrees)));

        CHECK(m.degree(max_degree_vertex) == bio_celegans_max_degree);
        CHECK(m.edges_count() == bio_celegans_edge_count);
        CHECK(m.vertices_count() == bio_celegans_vertex_count);
    }
}

TEST_CASE("Graph Reading, Aves Songbird Social") {
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

    // we start at 0 so we must have 117 + 1 vertices, the vertex count differs
    // here since all vertices must be stored starting from 0 to highest vertex
    // ID which in this case is 117
    unsigned int constexpr aves_songbird_social_vertex_count = 117 + 1;
    unsigned int constexpr aves_songbird_social_edge_count = 1027 * 2;
    dhb::Degree constexpr aves_songbird_social_max_degree = 56;
    std::vector<dhb::Degree> extracted_degrees = dhb::degrees_from(edges);

    SECTION("read data") {
        CHECK(edges.size() == aves_songbird_social_edge_count);
        CHECK(dhb::graph::vertex_count(edges) == aves_songbird_social_vertex_count);
        dhb::Degree const max_degree =
            *std::max_element(std::begin(extracted_degrees), std::end(extracted_degrees));

        CHECK(max_degree == aves_songbird_social_max_degree);
    }

    SECTION("initializing DHB matrix") {
        dhb::Matrix<sr::Weight> graph_init(vertex_count);
        for (auto e : edges) {
            graph_init.insert(e.source, e.target.vertex, e.target.data.weight);
        }
        dhb::Matrix<sr::Weight> const& graph = graph_init;

        CHECK(graph.vertices_count() == aves_songbird_social_vertex_count);

        size_t const max_degree_vertex = std::distance(
            std::begin(extracted_degrees),
            std::max_element(std::begin(extracted_degrees), std::end(extracted_degrees)));

        CHECK(graph.degree(max_degree_vertex) == aves_songbird_social_max_degree);
        CHECK(graph.edges_count() == aves_songbird_social_edge_count);
        CHECK(graph.vertices_count() == aves_songbird_social_vertex_count);
    }

    SECTION("sample of edge weight is correct") {
        dhb::Matrix<sr::Weight> graph_init(vertex_count);
        for (auto e : edges) {
            graph_init.insert(e.source, e.target.vertex, e.target.data.weight);
        }
        dhb::Matrix<sr::Weight> const& graph = graph_init;

        dhb::Vertex v_97 = 97;
        dhb::Vertex v_98 = 98;
        constexpr float weight_97_to_98 = 0.0028388928318f;
        CHECK(weight_97_to_98 == Catch::Approx(graph.neighbors(v_97).iterator_to(v_98)->data()));
    }
}

TEST_CASE("Graph Reading, Power 494 Bus") {
    dhb::EdgeID constexpr default_id = 0u;
    dhb::Edges edges;
    auto emplace = [&](dhb::Vertex const u, dhb::Vertex const v, dhb::Weight const w) {
        edges.push_back(dhb::Edge{u, dhb::Target{v, dhb::EdgeData{w, default_id}}});
    };

    std::ifstream power_494_bus_input(power_494_bus_graph_path);
    auto const [vertex_count, edge_count] =
        gdsb::read_graph<dhb::Vertex, decltype(emplace),
                         gdsb::MatrixMarketUndirectedWeightedNoLoopStatic>(power_494_bus_input,
                                                                           std::move(emplace));

    uint64_t constexpr power_494_bus_edge_count_original = 1080u * 2u;

    // + 1 due to missing vertex 0
    unsigned int constexpr power_494_bus_vertex_count = 494u + 1u;
    // undirected, removed self loops
    REQUIRE(power_494_bus_vertex_count == vertex_count);
    unsigned int constexpr power_494_bus_edge_count = 586u * 2u;
    REQUIRE(power_494_bus_edge_count == edge_count);

    SECTION("read data") {
        REQUIRE(power_494_bus_edge_count == edges.size());

        CHECK(dhb::graph::vertex_count(edges) == power_494_bus_vertex_count);
    }

    SECTION("initializing DHB matrix") {
        dhb::Matrix<sr::Weight> graph_init(vertex_count);
        for (auto e : edges) {
            graph_init.insert(e.source, e.target.vertex, e.target.data.weight);
        }
        dhb::Matrix<sr::Weight> const& graph = graph_init;

        CHECK(graph.vertices_count() == power_494_bus_vertex_count);

        CHECK(graph.edges_count() == edges.size());
        CHECK(graph.vertices_count() == power_494_bus_vertex_count);
    }
}

TEST_CASE("Graph Reading, Aves Thornbill Farine") {
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

    // + 1 due to missing vertex 0
    unsigned int constexpr aves_thornbill_farine_vertex_count = 62u + 1u;
    // undirected
    unsigned int constexpr aves_thornbill_farine_edge_count = 1151u * 2u;

    SECTION("read data") {
        CHECK(aves_thornbill_farine_vertex_count == vertex_count);
        CHECK(aves_thornbill_farine_edge_count == edges.size());

        CHECK(dhb::graph::vertex_count(edges) == aves_thornbill_farine_vertex_count);
    }

    SECTION("initializing DHB matrix") {
        dhb::Matrix<sr::Weight> graph_init(vertex_count);
        for (auto e : edges) {
            graph_init.insert(e.source, e.target.vertex, e.target.data.weight);
        }
        dhb::Matrix<sr::Weight> const& graph = graph_init;
        CHECK(graph.vertices_count() == aves_thornbill_farine_vertex_count);

        CHECK(graph.edges_count() == edges.size());
        CHECK(graph.vertices_count() == aves_thornbill_farine_vertex_count);
    }
}