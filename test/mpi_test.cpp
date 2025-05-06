#include <catch2/catch_test_macros.hpp>

#include <scalerunner/mpi_utils.h>

#include <gdsb/mpi_graph_io.h>

#include <filesystem>

TEST_CASE("MPI, sanity check") {
    CHECK(sr::mpi::com_rank() != -1);
    CHECK(sr::mpi::com_size() > 0);
}

TEST_CASE("MPI, file_offset()")
{
    SECTION("simulating 2 MPI processes") {
        std::vector<uint64_t> paths_sizes = {32u, 245u};

        MPI_Offset offset_rank_0 = 0u;
        MPI_Offset offset_rank_1 = 32u * sizeof(dhb::Vertex);

        CHECK(sr::mpi::file_offset(paths_sizes, 0u) == offset_rank_0);
        CHECK(sr::mpi::file_offset(paths_sizes, 1u) == offset_rank_1);
    }

    SECTION("simulating 4 MPI processes") {
        std::vector<uint64_t> paths_sizes = {32u, 245u, 532u, 43};

        MPI_Offset offset_rank_0 = 0u;
        MPI_Offset offset_rank_1 = paths_sizes[0] * sizeof(dhb::Vertex);
        MPI_Offset offset_rank_2 = offset_rank_1 + (paths_sizes[1] * sizeof(dhb::Vertex));
        MPI_Offset offset_rank_3 = offset_rank_2 + (paths_sizes[2] * sizeof(dhb::Vertex));

        CHECK(sr::mpi::file_offset(paths_sizes, 0u) == offset_rank_0);
        CHECK(sr::mpi::file_offset(paths_sizes, 1u) == offset_rank_1);
        CHECK(sr::mpi::file_offset(paths_sizes, 2u) == offset_rank_2);
        CHECK(sr::mpi::file_offset(paths_sizes, 3u) == offset_rank_3);
    }
}

TEST_CASE("MPI, determine_paths_sizes()") {
    size_t constexpr count = 4;
    size_t constexpr length = 4;
    sr::ConsecutivePaths paths(count, length);

    std::vector<uint64_t> paths_sizes = sr::mpi::determine_paths_sizes(paths);

    REQUIRE(paths_sizes.size() == sr::mpi::com_size());

    if (sr::mpi::com_size() == 2) {
        REQUIRE(paths_sizes.size() == 2);
    }

    size_t constexpr total_path_data_length = count * length;
    for (auto ps : paths_sizes) {
        CHECK(ps == total_path_data_length);
    }
}

TEST_CASE("MPI", "[stream]") {
    size_t constexpr count = 4;
    size_t constexpr length = 4;
    sr::ConsecutivePaths paths(count, length);
    size_t const paths_total_length = paths.vector().size();
    CHECK(paths_total_length == count * length);

    sr::Path path1{1, 2, 3, 4};
    sr::Path path2{5, 6, 7, 8};
    sr::Path path3{9, 10, 11, 12};
    sr::Path path4{13, 14, 15, 16};

    REQUIRE(paths.insert(0, path1) != std::end(paths.vector()));
    REQUIRE(paths.insert(1, path2) != std::end(paths.vector()));
    REQUIRE(paths.insert(2, path3) != std::end(paths.vector()));
    REQUIRE(paths.insert(3, path4) != std::end(paths.vector()));

    std::string const path_output_filename = "paths_stream_test_file.sr";

    std::filesystem::path file_path(path_output_filename);
    SECTION("stream file") {
        gdsb::mpi::FileWrapper file(file_path, true, 0, MPI_MODE_CREATE | MPI_MODE_WRONLY);
        sr::mpi::stream(file.get(), paths);
    }

    SECTION("restore file to paths on root") {
        if (sr::mpi::is_root()) {
            std::ifstream test_file;
            test_file.open(path_output_filename.c_str(), std::ios::binary);

            size_t const total_count = count * sr::mpi::com_size();
            sr::ConsecutivePaths read_paths(total_count, length);
            size_t const read_paths_total_length = paths_total_length * sr::mpi::com_size();
            REQUIRE(read_paths.vector().size() == read_paths_total_length);

            dhb::Vertex* read_paths_pointer = read_paths.vector().data();

            for (size_t i = 0u; i < read_paths_total_length; ++i) {
                test_file.read(reinterpret_cast<char*>(&(read_paths_pointer[i])), sizeof(uint32_t));
            }

            std::vector<dhb::Vertex> expected_vector;
            for (int i = 0; i < sr::mpi::com_size(); ++i) {
                expected_vector.insert(std::end(expected_vector), std::begin(paths.vector()),
                                       std::end(paths.vector()));
            }

            CHECK(expected_vector == read_paths.vector());
        }
    }

    bool constexpr delete_file = true;
    SECTION("delete file on root") {
        if (sr::mpi::is_root() and delete_file) {
            REQUIRE(std::remove(file_path.c_str()) == 0);
        }
    }
}
