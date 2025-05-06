#include <catch2/catch_test_macros.hpp>

#include <scalerunner/random_walk.h>

TEST_CASE("ConsecutivePaths, fundamentals") {
    size_t constexpr size = 8;
    size_t constexpr length = 4;
    sr::ConsecutivePaths cp(8, 4);

    SECTION("first and last path") {
        REQUIRE(cp[0] == dhb::invalidVertex());
        REQUIRE(cp[size - 1] == dhb::invalidVertex());
    }

    SECTION("begin_of") {
        sr::Path::iterator begin = cp.begin_of(0u);
        REQUIRE(*begin == dhb::invalidVertex());
    }

    SECTION("insert, index 0") {
        size_t constexpr index = 0u;
        sr::Path const new_path = {1u, 2u, 3u, 4u};
        auto path_it = cp.insert(index, new_path);

        CHECK(*path_it++ == 1);
        CHECK(*path_it++ == 2);
        CHECK(*path_it++ == 3);
        CHECK(*path_it++ == 4);
    }

    SECTION("insert, index 0, path not full") {
        size_t constexpr index = 0u;
        sr::Path const new_path = {1u, 2u, 3u};
        auto path_it = cp.insert(index, new_path);

        CHECK(*path_it++ == 1);
        CHECK(*path_it++ == 2);
        CHECK(*path_it++ == 3);
        CHECK(*path_it++ == dhb::invalidVertex());
    }

    SECTION("insert, 4 paths") {
        size_t constexpr size = 4;
        size_t constexpr length = 4;
        sr::ConsecutivePaths paths(8, 4);

        sr::Path path1 = {1, 2, 3, 4};
        sr::Path path2 = {5, 6, 7, 8};
        sr::Path path3 = {9, 10, 11, 12};
        sr::Path path4 = {13, 14, 15, 16};

        auto it = paths.insert(0, path1);
        REQUIRE(it != paths.vector().end());
        REQUIRE(path1.size() == 4u);
        CHECK(*it++ == path1[0]);
        CHECK(*it++ == path1[1]);
        CHECK(*it++ == path1[2]);
        CHECK(*it == path1[3]);

        it = paths.insert(1, path2);
        REQUIRE(it != paths.vector().end());
        REQUIRE(path2.size() == 4u);
        CHECK(*it++ == path2[0]);
        CHECK(*it++ == path2[1]);
        CHECK(*it++ == path2[2]);
        CHECK(*it == path2[3]);

        it = paths.insert(2, path3);
        REQUIRE(it != paths.vector().end());
        REQUIRE(path3.size() == 4u);
        CHECK(*it++ == path3[0]);
        CHECK(*it++ == path3[1]);
        CHECK(*it++ == path3[2]);
        CHECK(*it == path3[3]);

        it = paths.insert(3, path4);
        REQUIRE(it != paths.vector().end());
        REQUIRE(path4.size() == 4u);
        CHECK(*it++ == path4[0]);
        CHECK(*it++ == path4[1]);
        CHECK(*it++ == path4[2]);
        CHECK(*it == path4[3]);
    }
}