#pragma once

#include <gdsb/graph_input.h>
#include <scalerunner/first_order_rw.h>
#include <scalerunner/utils.h>

#include <omp.h>

#include <cassert>
#include <vector>

namespace sr {

template <class RWAlgorithm> Paths schedule(RWAlgorithm&& rw_algorithm, size_t const count) {
    Paths paths;

    for (size_t i = 0; i < count; ++i) {
        paths.emplace_back(rw_algorithm());
    }

    return paths;
}

template <typename RWAlgorithmF, typename StartVertexF, class ThreadPrivateData>
Paths schedule_task_based(RWAlgorithmF&& rw_algorithm, StartVertexF&& start_vertex_f,
                          size_t const count, std::vector<ThreadPrivateData>& t_data) {
    assert(t_data.size() == omp_get_max_threads());
    Paths paths;
    paths.resize(count);

#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp taskloop
            for (size_t round = 0; round < count; ++round) {
                paths[round] = rw_algorithm(start_vertex_f(round), t_data[omp_get_thread_num()]);
            }
        }
    }

    return paths;
}

template <typename RWAlgorithmF, typename StartVertexF, class ThreadPrivateData>
ConsecutivePaths schedule(RWAlgorithmF&& rw_algorithm, StartVertexF&& start_vertex_f,
                          size_t const count, size_t const length,
                          std::vector<ThreadPrivateData>& t_data) {
    assert(t_data.size() == omp_get_max_threads());
    ConsecutivePaths paths(count, length);

#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp taskloop
            for (size_t round = 0; round < count; ++round) {
                auto path_begin = paths.begin_of(round);
                rw_algorithm(start_vertex_f(round), path_begin, t_data[omp_get_thread_num()]);
            }
        }
    }

    return paths;
}

} // namespace sr
