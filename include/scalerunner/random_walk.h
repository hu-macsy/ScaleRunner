#pragma once

#include <dhb/graph.h>

#include <cassert>
#include <type_traits>
#include <vector>

namespace sr {

using Path = std::vector<dhb::Vertex>;
using Paths = std::vector<Path>;

class ConsecutivePaths {
  public:
    ConsecutivePaths(size_t count, size_t length)
        : m_count(count), m_length(length), m_paths(count * length, dhb::invalidVertex()) {}

    using OpParReturnType = std::vector<dhb::Vertex>::reference;

    OpParReturnType operator[](size_t const idx) {
        assert(idx < m_count);
        size_t const offset = m_length * idx;
        return m_paths[offset];
    }

    Path::iterator insert(size_t const idx, Path&& path) {
        assert(path.size() <= m_length);
        auto stored_path_begin = begin_of(idx);
        auto stored_path_end = end_of(idx);
        auto stored_it = stored_path_begin;

        for (auto v = std::begin(path); v != std::end(path) && stored_it != stored_path_end;
             ++v, ++stored_it) {
            *stored_it = *v;
        }

        return stored_path_begin;
    }

    Path::iterator insert(size_t const idx, Path const& path) {
        assert(path.size() <= m_length);
        auto stored_path_begin = begin_of(idx);
        auto stored_path_end = end_of(idx);
        auto stored_it = stored_path_begin;

        for (auto v = std::begin(path); v != std::end(path) && stored_it != stored_path_end;
             ++v, ++stored_it) {
            *stored_it = *v;
        }

        return stored_path_begin;
    }

    Path::iterator begin_of(size_t const path_idx) {
        assert(path_idx < m_count);
        size_t const offset = m_length * path_idx;
        auto it = std::begin(m_paths);
        std::advance(it, offset);
        return it;
    }

    Path::const_iterator end_of(size_t const idx) const {
        assert(idx < m_count);
        size_t const offset = m_length * idx + m_length;
        auto it = std::cbegin(m_paths);
        std::advance(it, offset);
        return it;
    }

    std::vector<dhb::Vertex> const& vector() const { return m_paths; }
    std::vector<dhb::Vertex>& vector() { return m_paths; }

  private:
    size_t m_count;
    size_t m_length;
    std::vector<dhb::Vertex> m_paths;
};
}
