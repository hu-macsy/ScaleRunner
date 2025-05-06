#include <catch2/catch_test_macros.hpp>

#include "graph.h"

#include <scalerunner/graph.h>
#include <scalerunner/rw_scheduler.h>
#include <scalerunner/utils.h>

#include <gdsb/graph_input.h>

#include <istream>

TEST_CASE("Scheduler") {
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

    SECTION("schedule 7 random walks, length 10") {
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

        auto rw_algorithm = [&m, &length, &rng, &step_f]() {
            return sr::first_order::walk(std::move(step_f), m, 1, length);
        };

        constexpr size_t rw_count = 7;
        sr::Paths paths = sr::schedule(std::move(rw_algorithm), rw_count);

        CHECK(paths.size() == rw_count);
    }
}

TEST_CASE("Task Scheduler") {
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

    SECTION("first test, 4 threads") {
        unsigned int const thread_count = 4;
        omp_set_num_threads(thread_count);
        REQUIRE(omp_get_max_threads() == thread_count);

        dhb::Matrix<sr::Weight> m(vertex_count);
        for (auto e : edges) {
            m.insert(e.source, e.target.vertex, e.target.data.weight);
        }

        constexpr size_t length = 10;

        std::vector<sr::RandomNumberGenPack> rngs(omp_get_max_threads());

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            return sr::first_order::step(n, rngs[omp_get_thread_num()].i_rng);
        };

        auto rw_algorithm = [&m, &length, &step_f](dhb::Vertex const u,
                                                   sr::RandomNumberGenPack& rng) {
            return sr::first_order::walk(std::move(step_f), m, u, length);
        };

        auto start_vertex_f = [&](size_t const walk_id) { return walk_id % m.vertices_count(); };

        constexpr size_t rw_count = 50;
        sr::Paths paths = sr::schedule_task_based(std::move(rw_algorithm),
                                                  std::move(start_vertex_f), rw_count, rngs);

        CHECK(paths.size() == rw_count);
    }
}

TEST_CASE("Scheduler Consecutive Path") {
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

    // TODO: deterministic paths
    SECTION("schedule 7 random walks, length 10") {
        unsigned int const thread_count = 4;
        omp_set_num_threads(thread_count);
        REQUIRE(omp_get_max_threads() == thread_count);

        dhb::Matrix<sr::Weight> m(vertex_count);
        for (auto e : edges) {
            m.insert(e.source, e.target.vertex, e.target.data.weight);
        }

        constexpr size_t length = 10;

        std::vector<sr::RandomNumberGenPack> rngs(omp_get_max_threads());

        auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n) {
            return sr::first_order::step(n, rngs[omp_get_thread_num()].i_rng);
        };

        auto rw_algorithm = [&m, &length, &step_f](dhb::Vertex const u, sr::Path::iterator begin,
                                                   sr::RandomNumberGenPack& rng) {
            sr::first_order::walk(std::move(step_f), m, u, begin, length);
        };

        auto start_vertex_f = [&](size_t const walk_id) { return walk_id % m.vertices_count(); };

        constexpr size_t rw_count = 50;
        sr::ConsecutivePaths paths = sr::schedule(
            std::move(rw_algorithm), std::move(start_vertex_f), rw_count, length, rngs);

        CHECK(paths.vector().size() == rw_count * length);
    }
}