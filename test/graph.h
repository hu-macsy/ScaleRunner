#pragma once

#include <dhb/graph.h>

#include <algorithm>
#include <string>

static std::string test_graph_path =
#ifdef SR_TEST_GRAPH_DIR
    std::string(SR_TEST_GRAPH_DIR) + "/";
#else
    "test/graphs/";
#endif

static std::string bio_celegans_graph_path{test_graph_path + "bio-celegans.mtx"};
static std::string aves_songbird_social_graph_path{test_graph_path + "aves-songbird-social.edges"};
static std::string power_494_bus_graph_path{test_graph_path + "power-494-bus.mtx"};
static std::string aves_thornbill_farine_graph_path{test_graph_path +
                                                    "aves-thornbill-farine.edges"};
static std::string hu_graph_path{test_graph_path + "hu-webgraph.edges"};