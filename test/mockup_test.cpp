#include <catch2/catch_test_macros.hpp>

#include "mockup.h"

#include <vector>

TEST_CASE("SeriesT") {
    SECTION("iterating over series") {
        std::vector<uint32_t> series{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u};
        std::vector<uint32_t> series_copy = series;
        SeriesT<uint32_t> engine{std::move(series)};

        auto series_it = std::begin(series_copy);
        auto series_end = std::end(series_copy);
        REQUIRE(series_it != series_end);

        for (; series_it != series_end; ++series_it) {
            CHECK(*series_it == engine());
        }

        REQUIRE(series_it == series_end);
    }

    SECTION("reach end of series") {
        std::vector<uint32_t> series{0u, 1u};
        auto series_copy = series;
        SeriesT<uint32_t> engine{std::move(series)};

        auto series_it = std::begin(series_copy);
        auto series_end = std::end(series_copy);
        REQUIRE(series_it != series_end);

        CHECK(*series_it++ == engine());
        CHECK(*series_it == engine());
        series_it = std::begin(series_copy);

        REQUIRE(*series_it++ == engine());
        REQUIRE(*series_it == engine());
    }

    SECTION("boolean series") {
        std::vector<bool> series{false, true, true, false, false, false, false, true, false};
        auto series_copy = series;

        SeriesT<bool> engine{std::move(series)};
        auto series_it = std::begin(series_copy);
        auto series_end = std::end(series_copy);
        REQUIRE(series_it != series_end);

        for (; series_it != series_end; ++series_it) {
            CHECK(*series_it == engine());
        }

        REQUIRE(series_it == series_end);
    }
}