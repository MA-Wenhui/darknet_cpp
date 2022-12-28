#ifndef LAYER_H
#define LAYER_H

#include <functional>

#include "activation.h"
#include "tensor.h"

typedef enum {
  CONVOLUTIONAL,
  DECONVOLUTIONAL,
  CONNECTED,
  MAXPOOL,
  SOFTMAX,
  DETECTION,
  DROPOUT,
  CROP,
  ROUTE,
  COST,
  NORMALIZATION,
  AVGPOOL,
  LOCAL,
  SHORTCUT,
  ACTIVE,
  RNN,
  GRU,
  LSTM,
  CRNN,
  BATCHNORM,
  NETWORK,
  XNOR,
  REGION,
  YOLO,
  ISEG,
  REORG,
  UPSAMPLE,
  LOGXENT,
  L2NORM,
  BLANK
} LAYER_TYPE;

typedef enum { SSE, MASKED, L1, SEG, SMOOTH, WGAN } COST_TYPE;

typedef struct {
  int batch;
  float learning_rate;
  float momentum;
  float decay;
  int adam;
  float B1;
  float B2;
  float eps;
  int t;
} update_args;

LAYER_TYPE string2LayerType(std::string type) {
  if (type == "shortcut") return SHORTCUT;
  if (type == "crop") return CROP;
  if (type == "cost") return COST;
  if (type == "detection") return DETECTION;
  if (type == "region") return REGION;
  if (type == "yolo") return YOLO;
  if (type == "iseg") return ISEG;
  if (type == "local") return LOCAL;
  if (type == "conv" || (type == "convolutional")) return CONVOLUTIONAL;
  if (type == "deconv]" || type == "deconvolutional") return DECONVOLUTIONAL;
  if (type == "activation") return ACTIVE;
  if (type == "logistic") return LOGXENT;
  if (type == "l2norm") return L2NORM;
  if (type == "net" || type == "network") return NETWORK;
  if (type == "crnn") return CRNN;
  if (type == "gru") return GRU;
  if (type == "lstm]") return LSTM;
  if (type == "rnn") return RNN;
  if (type == "conn]" || type == "connected") return CONNECTED;
  if (type == "max]" || type == "maxpool") return MAXPOOL;
  if (type == "reorg") return REORG;
  if (type == "avg" || type == "avgpool") return AVGPOOL;
  if (type == "dropout") return DROPOUT;
  if (type == "lrn" || type == "normalization") return NORMALIZATION;
  if (type == "batchnorm") return BATCHNORM;
  if (type == "soft" || type == "softmax") return SOFTMAX;
  if (type == "route") return ROUTE;
  if (type == "upsample") return UPSAMPLE;
  return BLANK;
}

class Layer {
 public:
  Layer(int batch, int h, int w, int c) : batch_(batch), h_(h), w_(w), c_(c) {}
  ~Layer() {}
  virtual void forward(Tensor<float>& input, Tensor<float>& output,
                       std::map<int, Tensor<float>> tensor_map) = 0;
  virtual void backward() = 0;
  virtual void load_weights(std::fstream& fs) {}

 public:
  LAYER_TYPE type_;
  std::string type_str_;
  int index_;
  ACTIVATION activation_;
  COST_TYPE cost_type_;

  int batch_;
  int h_;
  int w_;
  int c_;

  int out_w_;
  int out_h_;
  int out_c_;

  uint64_t output_length_ = 0;
  uint64_t input_length_ = 0;
};

using LayerPtr = std::shared_ptr<Layer>;
#endif