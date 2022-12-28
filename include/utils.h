#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <vector>
#include <string>

#define TERMINAL_COLOR_RED "\033[0;31m"
#define TERMINAL_COLOR_GREEN "\033[0;32m"
#define TERMINAL_COLOR_BLUE "\033[0;34m"
#define TERMINAL_COLOR_YELLOW "\033[0;33m"
#define TERMINAL_COLOR_RESET "\033[0m"

template<typename T>
std::string join(std::vector<T> const &vec, std::string delim)
{
    if (vec.empty()) {
        return std::string();
    }
 
    return std::accumulate(vec.begin() + 1, vec.end(), std::to_string(vec[0]),
                [&](const std::string& a, T b){
                    return a + delim + std::to_string(b);
                });
}

static std::random_device rd{};
static std::mt19937 gen{rd()};

template<class T>
T rand_normal(float mean,float dev){
  std::normal_distribution<T> d{mean, dev};
  return d(gen);
}


#endif