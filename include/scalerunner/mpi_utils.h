#pragma once

#include <scalerunner/random_walk.h>

#include <mpi.h>

namespace sr {

namespace mpi {

int com_rank(MPI_Comm communicator = MPI_COMM_WORLD);

int com_size(MPI_Comm communicator = MPI_COMM_WORLD);

bool is_root(MPI_Comm communicator = MPI_COMM_WORLD);

MPI_Offset file_offset(std::vector<uint64_t> const& paths_sizes, uint32_t const my_mpi_rank);

std::vector<uint64_t> determine_paths_sizes(ConsecutivePaths const&);

void stream(MPI_File, ConsecutivePaths&);

} // namespace mpi

} // namespace sr