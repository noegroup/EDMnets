//
// Created by mho on 7/8/19.
//

#pragma once

#include <cstddef>
#include <vector>
#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "index.h"
#include "random.h"
#include "asp.h"

namespace py = pybind11;

template<typename T>
using np_array = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;

namespace edmnets::hungarian {

template<typename fp>
class Array2DAccess {
public:

    Array2DAccess(fp* arr, std::size_t n) : arr(arr), n(n) {}

    fp& operator()(std::size_t i, std::size_t j) {
        return *(arr + i * n + j);
    }

    fp operator()(std::size_t i, std::size_t j) const {
        return *(arr + i * n + j);
    }

private:
    fp* arr;
    std::size_t n;
};

template<typename fp>
class Masked2DAccess {
public:

    Masked2DAccess(fp* arr, std::size_t nTotal, std::vector<std::size_t> offsets)
            : arr(arr), nTotal(nTotal), offsets(std::move(offsets)) {}

    fp& operator()(std::size_t i, std::size_t j) {
        return *(arr + offsets.at(i) * nTotal + offsets.at(j));
    }

    fp operator()(std::size_t i, std::size_t j) const {
        return *(arr + offsets.at(i) * nTotal + offsets.at(j));
    }

private:
    std::size_t nTotal;
    fp* arr;
    std::vector<std::size_t> offsets;
};

template<typename fp>
std::vector<std::decay_t<fp>> costMatrix(fp* a, fp* b, std::size_t n) {
    using ResultType = std::vector<std::decay_t<fp>>;
    ResultType result (static_cast<typename ResultType::size_type>(n*n), std::numeric_limits<std::decay_t<fp>>::max());

    edmnets::index::Index2D ix (n, n);

    auto rowSum = [ix, n](fp* arr, std::size_t row) -> typename ResultType::value_type {
        auto beginIx = ix(row, static_cast<std::size_t>(0));
        return std::accumulate(arr + beginIx, arr + beginIx + n, 0);
    };

    auto it = result.begin();
    for(std::size_t i= 0; i < n; ++i) {
        for(std::size_t j = 0; j < n; ++j) {

            auto cost = std::abs(rowSum(a, i) - rowSum(b, j));
            *it = cost;
            ++it;
        }
    }

    return result;
}

}
