#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include "graph.h"
#include "mockup.h"

#include <scalerunner/node2vec.h>
#include <scalerunner/second_order_rw.h>
#include <scalerunner/utils.h>

#include <dhb/dynamic_hashed_blocks.h>
#include <gdsb/experiment.h>
#include <gdsb/graph_input.h>

#include <istream>
#include <optional>

TEST_CASE("2nd Order RW") {
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

    SECTION("Simple 2nd Order") {
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

        auto step_f =
            [&i_rng](dhb::Matrix<sr::Weight>::ConstNeighborView n,
                     std::optional<dhb::Vertex> last_vertex) -> std::optional<dhb::Vertex> {
            bool n_found = false;
            dhb::Vertex v{0u};
            do {
                size_t index = i_rng(n.degree() - 1);
                auto candidate = n.begin();
                std::advance(candidate, index);
                if (!last_vertex || candidate->vertex() != last_vertex || n.degree() == 1) {
                    v = candidate->vertex();
                    n_found = true;
                }
            } while (!n_found);

            if (!n_found) {
                return {};
            }

            return v;
        };

        sr::Path const path = sr::second_order(std::move(step_f), graph, start_vertex, length);

        REQUIRE(path.size() == length);
    }
}