//
// Created by mho on 4/1/19.
//

#pragma once

#include <random>
#include <thread>

namespace edmnets::random {

template<typename RealType, typename Generator = std::default_random_engine>
RealType normal(const RealType mean = 0.0, const RealType variance = 1.0) {
    static thread_local Generator generator(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
    std::normal_distribution<RealType> distribution(mean, variance);
    return distribution(generator);
}

template<typename RealType, typename Generator = std::default_random_engine>
RealType uniform_real(const RealType a = 0.0, const RealType b = 1.0) {
    static thread_local Generator generator(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
    std::uniform_real_distribution<RealType> distribution(a, b);
    return distribution(generator);
}

template<typename IntType, typename Generator = std::default_random_engine>
IntType uniform_int(const IntType a, const IntType b) {
    static thread_local Generator generator(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
    std::uniform_int_distribution<IntType> distribution(a, b);
    return distribution(generator);
}

template<typename RealType, typename Generator = std::default_random_engine>
RealType exponential(RealType lambda = 1.0) {
    static thread_local Generator generator(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
    std::exponential_distribution<RealType> distribution(lambda);
    return distribution(generator);
}

template<typename Iter, typename Gen = std::default_random_engine>
Iter random_element(Iter start, const Iter end) {
    using IntType = typename std::iterator_traits<Iter>::difference_type;
    std::advance(start, uniform_int<IntType, Gen>(0, std::distance(start, end)));
    return start;
}

}
