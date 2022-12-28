#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <string_view>
#include "spdlog/spdlog.h"
#include "toml/toml.hpp"
#include "utils.h"

using namespace std::string_view_literals;

std::vector<int> get_config_vec_int(toml::table config,
                                    std::string_view key) {
  std::vector<int> ret;
  try {
    auto arr = config[key].as_array();
    if (arr) {
      for (auto &&elem : *arr) {
        ret.push_back(elem.as_integer()->get());
      }
    } else {
      throw std::runtime_error(
          std::string("get_config failed to get ").append(key));
    }
    return ret;
  } catch (std::bad_optional_access) {
    throw std::runtime_error(
        std::string("get_config failed to get ").append(key));
  }
}


std::vector<float> get_config_vec_float(
    toml::table config, std::string_view key) {
  std::vector<float> ret;
  try {
    auto arr = config[key].as_array();

    if (arr) {
      for (auto &&elem : *arr) {
        ret.push_back(elem.as_floating_point()->get());
      }
    } else {
      throw std::runtime_error(
          std::string("get_config failed to get ").append(key));
    }
    return ret;
  } catch (std::bad_optional_access) {
    throw std::runtime_error(
        std::string("get_config failed to get ").append(key));
  }
}

template <typename T>
T get_config(toml::table config, std::string_view key) {
  try {
    return config[key].value<T>().value();
  } catch (std::bad_optional_access) {
    throw std::runtime_error(
        std::string("get_config failed to get ").append(key));
  }
}

template <typename T>
T get_config(toml::table config, std::string_view key,T default_value) {
  try {
    return config[key].value<T>().value();
  } catch (std::bad_optional_access) {
    SPDLOG_WARN("get_config get {} as default: {}", key, default_value);
    return default_value;
  }
}

template <typename T>
T get_config(toml::v3::node_view<toml::v3::node> config, std::string_view key) {
  try {
    return config[key].value<T>().value();
  } catch (std::bad_optional_access) {
    throw std::runtime_error(
        std::string("get_config failed to get ").append(key));
  }
}

template <typename T>
T get_config(toml::v3::node_view<toml::v3::node> config, std::string_view key,
             T default_value) {
  try {
    return config[key].value<T>().value();
  } catch (std::bad_optional_access) {
    SPDLOG_WARN("get_config get {} as default: {}", key, default_value);
    return default_value;
  }
}

#endif