#pragma once

#include <dhb/dynamic_hashed_blocks.h>
#include <scalerunner/graph.h>

#include <vector>

template <typename T> class SeriesT {
  public:
    SeriesT() = delete;

    SeriesT(std::vector<T>&& s)
        : m_series(std::move(s)), m_series_it(std::cbegin(m_series)),
          m_series_last(std::prev(std::cend(m_series))) {
        assert(m_series_it != std::cend(m_series));
    }

    ~SeriesT() = default;

    T operator()() {
        if (m_series_it == m_series_last) {
            m_series_it = std::cbegin(m_series);
            return *m_series_last;
        }

        return *m_series_it++;
    }

    // interface to imitate RandomEngine
    T operator()(T const) { return this->operator()(); }

  private:
    std::vector<T> const m_series;
    typename std::vector<T>::const_iterator m_series_it;
    typename std::vector<T>::const_iterator m_series_last;
};

inline std::vector<size_t> construct_indices(dhb::Matrix<sr::Weight> const& m,
                                             std::vector<dhb::Vertex> const& series,
                                             dhb::Vertex v_start) {
    std::vector<size_t> indices;
    dhb::Vertex u = v_start;
    for (size_t i = 0; i < series.size(); ++i) {
        auto n = m.neighbors(u);
        size_t index = 0;
        for (auto it = n.begin(); it != n.end(); ++it, ++index) {
            if (it->vertex() == series[i]) {
                indices.emplace_back(index);
            }
        }
        u = series[i];
    }

    return indices;
};