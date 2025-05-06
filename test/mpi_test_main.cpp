#define CATCH_CONFIG_RUNNER

#include <catch2/catch_session.hpp>

#include <mpi.h>

#define MPI_COLLECT_OUTPUT 1

// The solution on how to use Catch2 for an MPI test environment is the result
// of following the ideas of the following thread:
// -
// https://stackoverflow.com/questions/58289895/is-it-possible-to-use-catch2-for-testing-an-mpi-code
//
//  Which points to:
// - https://github.com/catchorg/Catch2/issues/566
// - https://github.com/catchorg/Catch2/blob/devel/docs/own-main.md
//
// We write our own main into the Catch2 setup code and call the binary using:
// mpiexec -n 2 ./scalerunner_mpi_test
// to run the SR MPI test on 2 processes.
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    Catch::Session session; // There must be exactly one instance

#if defined(MPI_COLLECT_OUTPUT)
    std::stringstream stringstr;
    /* save old buffer and redirect output to string stream */
    auto cout_buf = std::cout.rdbuf(stringstr.rdbuf());
#endif

    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) { // Indicates a command line error
        MPI_Finalize();
        return returnCode;
    }

    int numFailed = session.run();

#if defined(MPI_COLLECT_OUTPUT)
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* reset buffer */
    std::cout.rdbuf(cout_buf);

    std::stringstream printRank;
    printRank << "Rank ";
    printRank.width(2);
    printRank << std::right << rank << ":\n";

    int size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 1; i < size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == rank) {
            /* if all tests are passed, it's enough if we hear that from
             * the master. Otherwise, print results */
            if (stringstr.str().rfind("All tests passed") == std::string::npos) {
                std::cout << printRank.str() + stringstr.str();
            }
        }
    }
    /* have master print last, because it's the one with the most assertions */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << printRank.str() + stringstr.str();
    }
#endif

    MPI_Finalize();
    return numFailed;
}
