
#include <scalerunner/node2vec.h>
#include <scalerunner/rw_scheduler.h>
#include <scalerunner/second_order_rw.h>
#include <scalerunner/utils.h>

#include <gdsb/graph_input.h>
#include <gdsb/timer.h>

#include <omp.h>

#include <fstream>
#include <iostream>

static std::string graph_name{"bio-celegans.mtx"};
static std::string graph_path =
#ifdef SR_TEST_GRAPH_DIR
    std::string(SR_TEST_GRAPH_DIR) + "/"
#else
    "test/graphs/"
#endif
    + graph_name;

int main(int argc, char** argv) {
    std::cout << "ScaleRunner Demo." << std::endl;

    unsigned int const thread_count = 4;
    omp_set_num_threads(thread_count);
    std::cout << "Running " << omp_get_max_threads() << " threads." << std::endl;

    dhb::Weight constexpr default_weight = 1.f;
    dhb::EdgeID constexpr default_id = 0u;
    dhb::Edges edges;
    auto emplace = [&](dhb::Vertex u, dhb::Vertex v) {
        edges.push_back(dhb::Edge{u, dhb::Target{v, dhb::EdgeData{default_weight, default_id}}});
    };
    std::ifstream bio_celegans_input(graph_path);
    auto const [vertex_count, edge_count] =
        gdsb::read_graph<dhb::Vertex, decltype(emplace),
                         gdsb::MatrixMarketUndirectedUnweightedNoLoopStatic>(bio_celegans_input,
                                                                             std::move(emplace));

    dhb::Matrix<sr::Weight> m(vertex_count);
    for (auto e : edges) {
        m.insert(e.source, e.target.vertex, e.target.data.weight);
    }

    std::cout << "Graph path: " << graph_path << std::endl;
    std::cout << "Graph n: " << m.vertices_count() << ", m: " << m.edges_count() << std::endl;

    constexpr size_t length = 10;
    constexpr dhb::Vertex start_vertex = 1;

    constexpr sr::node2vec::Parameter n2v_parameter{1.f, .5f};

    std::cout << "Running Node2Vec Algorithm with p (return): " << n2v_parameter.p_return
              << ", q (in/out): " << n2v_parameter.q_in_out << "." << std::endl;

    constexpr float travel_to_random_vertex_p = 0.1f;
    std::vector<sr::RandomNumberGenPack> rngs(omp_get_max_threads());

    auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n,
                      std::optional<dhb::Vertex> last_vertex) -> std::optional<dhb::Vertex> {
        auto chosen_vertex = sr::node2vec::unweighted::step_rejection_sampling(
            m, n, last_vertex, n2v_parameter, sr::node2vec::max_alpha(n2v_parameter),
            rngs[omp_get_thread_num()].i_rng, rngs[omp_get_thread_num()].f_rng);

        return chosen_vertex;
    };

    auto node2vec_rw = [&](dhb::Vertex const u, sr::RandomNumberGenPack&) {
        return sr::second_order(std::move(step_f), m, u, length);
    };

    auto start_vertex_f = [&](size_t const walk_id) { return walk_id % m.vertices_count(); };

    constexpr size_t rw_count = 4000000u;

    gdsb::WallTimer timer;

    timer.start();
    sr::Paths paths =
        sr::schedule_task_based(std::move(node2vec_rw), std::move(start_vertex_f), rw_count, rngs);
    timer.end();
    std::cout << "Time for RW: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(timer.duration()).count()
              << "\n";

    if (rw_count <= 50u) {

        std::cout << "Computed paths: " << paths.size() << "\n";
        // std::cout << "HIER2" << std::endl;

        for (size_t p = 0; p < paths.size(); ++p) {
            sr::Path const& path = paths[p];
            std::cout << "Path " << (p + 1) << " of size " << path.size() << ": [";
            for (size_t i = 0; i < path.size(); ++i) {
                std::cout << path[i];
                if (i < path.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]\n";
        }

        std::cout << std::endl;
    }

    bool write_to_file = false;

    if (write_to_file) {
        std::filesystem::path output_path{"computed_paths.txt"};
        std::cout << "Writing paths to file: " << output_path.string() << std::endl;

        std::fstream output_stream;
        output_stream.open(output_path, output_stream.out);

        if (!output_stream.is_open()) {
            std::cerr << "Path: " << output_path.string() << " could not be opened." << std::endl;
        }

        sr::stream(output_stream, paths);

        output_stream.flush();
    }

    return 0;
}