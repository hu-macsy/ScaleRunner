#include <scalerunner/mpi_utils.h>

#include <stdexcept>

int sr::mpi::com_rank(MPI_Comm communicator) {
    int rank;
    MPI_Comm_rank(communicator, &rank);
    return rank;
}

int sr::mpi::com_size(MPI_Comm communicator) {
    int size;
    MPI_Comm_size(communicator, &size);
    return size;
}

bool sr::mpi::is_root(MPI_Comm communicator) { return sr::mpi::com_rank(communicator) == 0; }

MPI_Offset sr::mpi::file_offset(std::vector<uint64_t> const& paths_sizes,
                              uint32_t const my_mpi_rank) {
    MPI_Offset offset = 0u;
    for (size_t i = 0; i < my_mpi_rank; ++i) {
        offset += paths_sizes[i];
    }
    offset = offset * sizeof(dhb::Vertex);

    return offset;
}

std::vector<uint64_t> sr::mpi::determine_paths_sizes(sr::ConsecutivePaths const& paths) {
    auto PathSizeMPIType = MPI_UINT64_T;

    uint64_t const my_paths_size = paths.vector().size();

    std::vector<uint64_t> paths_sizes(com_size(), 0u);

    int allgather_error = MPI_Allgather(&my_paths_size, 1, PathSizeMPIType, &(paths_sizes[0]), 1,
                                        PathSizeMPIType, MPI_COMM_WORLD);
    if (allgather_error != MPI_SUCCESS) {
        throw std::runtime_error("Could not gather respective sizes of paths.");
    }

    return paths_sizes;
}

// TODO: Currently based on 32bit vertices.
void sr::mpi::stream(MPI_File file, sr::ConsecutivePaths& paths) {
    std::vector<uint64_t> const paths_sizes = determine_paths_sizes(paths);
    MPI_Offset const my_offset = file_offset(paths_sizes, static_cast<uint32_t>(com_rank()));

    dhb::Vertex* raw_data = paths.vector().data();

    // TODO: MPI_File_write_at_all accepts only type int as count parameter. Our
    // paths per process can therefore never be greater than 2^31.
    int const raw_data_size = static_cast<int>(paths.vector().size());

    MPI_Status status;
    int file_write_all =
        MPI_File_write_at_all(file, my_offset, raw_data, raw_data_size, MPI_UINT32_T, &status);
    if (file_write_all != MPI_SUCCESS) {
        throw std::runtime_error("Could not write file using MPI routine.");
    }
}