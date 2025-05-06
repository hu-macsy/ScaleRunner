# ScaleRunner

ScaleRunner, a modular C++ library with full CMake integration using hybrid
parallelism (MPI+OpenMP) to execute first- and second-order random walks on
multi-CPU systems. Our neighborhood sampling API provides functionality for
both unweighted and weighted graphs. 

## Build

You will need to install the following libraries, provide the path during the
configure step, or you install them system wide:

- [DHB](https://github.com/hu-macsy/dhb)
- [OpenMP](https://www.openmp.org/)
- [MPI (e.g. MPICH implementation)](https://www.mpich.org/)

Once installed you can build ScaleRunner

```bash
mkdir build;
cd build;
cmake -GNinja ../
ninja
```

For installing ScaleRunner (system wide) use:

```bash
ninja install
```

## Demo

For building the demo set the CMake option `SR_DEMO=On` e.g.:

```bash
mkdir build;
cd build;
cmake -GNinja -DSR_DEMO=On ../
ninja
```

Execute the demo using:

```
./scalerunner_demo
```

Please see the [demo code](/demo/demo.cpp) for details on our demo
implementation.

## Tests

To build and run tests you will need:

- [Catch2](https://github.com/catchorg/Catch2)
- [GDSB](https://github.com/hu-macsy/graph-ds-benchmark)

[Catch2](https://github.com/catchorg/Catch2) is pulled in as a submodule and
must not be installed. Instead, if you forgot to clone this repository
recursively, you can use `git submodule init` to use the defined git submodule.

[GDSB](https://github.com/hu-macsy/graph-ds-benchmark) must be installed.

Once all dependencies are installed you can build ScaleRunner with tests using:

```bash
mkdir build;
cd build;
cmake -GNinja -DSR_BUILD_TESTS=On ../
ninja
```

Then you may either use `ctest` for running all tests or manually calling the
test binary:

```bash
./scalerunner_test
```

Additionally, we provide MPI tests of ScaleRunner. To enable these you must set
`SR_BUILD_MPI_TESTS=On` e.g.:

```bash
mkdir build;
cd build;
cmake -GNinja -DSR_BUILD_TESTS=On -DSR_BUILD_MPI_TESTS=On ../
ninja
```

For all the tests to compile successfully, as already done in this git project
one must checkout/switch to commit hash
`fa43b77429ba76c462b1898d6cd2f2d7a9416b14` within the Catch2 submodule. 

Execute the tests using an MPI runtime:

```bash
mpiexec -n 2 ./scalerunner_mpi_test
```

## Authors

| Name            | E-Mail                 | Affiliation |
|-----------------|------------------------|-------------|
| Florian Willich | f.willich@hu-berlin.de | HU Berlin   |

The ScaleRunner project was lead by [Henning
Meyerhenke](mailto:meyerhenke@kit.edu).