#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "common.h"

typedef enum {
  LOGISTIC,
  RELU,
  RELIE,
  LINEAR,
  RAMP,
  TANH,
  PLSE,
  LEAKY,
  ELU,
  LOGGY,
  STAIR,
  HARDTAN,
  LHTAN,
  SELU
} ACTIVATION;

ACTIVATION string2Activation(std::string_view s) {
  if (s == "logistic") return LOGISTIC;
  if (s == "loggy") return LOGGY;
  if (s == "relu") return RELU;
  if (s == "elu") return ELU;
  if (s == "selu") return SELU;
  if (s == "relie") return RELIE;
  if (s == "plse") return PLSE;
  if (s == "hardtan") return HARDTAN;
  if (s == "lhtan") return LHTAN;
  if (s == "linear") return LINEAR;
  if (s == "ramp") return RAMP;
  if (s == "leaky") return LEAKY;
  if (s == "tanh") return TANH;
  if (s == "stair") return STAIR;
  SPDLOG_WARN("Couldn't find activation function %s, going with ReLU\n", s);
  return RELU;
}

std::string activation2str(ACTIVATION a) {
  switch (a) {
    case LOGISTIC:
      return "logistic";
    case LOGGY:
      return "loggy";
    case RELU:
      return "relu";
    case ELU:
      return "elu";
    case SELU:
      return "selu";
    case RELIE:
      return "relie";
    case RAMP:
      return "ramp";
    case LINEAR:
      return "linear";
    case TANH:
      return "tanh";
    case PLSE:
      return "plse";
    case LEAKY:
      return "leaky";
    case STAIR:
      return "stair";
    case HARDTAN:
      return "hardtan";
    case LHTAN:
      return "lhtan";
    default:
      break;
  }
  return "relu";
}

float activate(float x, ACTIVATION a) {
  switch (a) {
    case LINEAR:
      return x;
    case LOGISTIC:
      return 1.0 / (1.0 + exp(-x));
    case RELU:
      return x > 0 ? x : 0;
    case LEAKY:
      return x > 0 ? x : 0.1 * x;
    case TANH:
      return (exp(2 * x) - 1) / (exp(2 * x) + 1);
    default:
      throw std::runtime_error("actication not implemented.");
  }
}

void activate_array(float *x, const int n, const ACTIVATION a) {
  int i;
  for (i = 0; i < n; ++i) {
    x[i] = activate(x[i], a);
  }
}


#endif