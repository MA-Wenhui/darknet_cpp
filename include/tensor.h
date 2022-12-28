#ifndef TENSOR_H
#define TENSOR_H

#include <iomanip>
#include <memory>

#include "utils.h"

enum TensorFormat {
  NCHW,  // RRRRR GGGGG BBBBB
  NHWC,  // RGB RGB RGB RGB RGB
};

template <typename T>
class Tensor {
 public:
 Tensor<T>& operator=(const Tensor<T> & in){
    batch_ = in.batch_;
    w_ = in.w_;
    h_ = in.h_;
    c_ = in.c_;
    if (in.data_size_ > data_size_) {
      data_.reset(new dtype[in.data_size_]);
    }
    memcpy(data_.get(), in.data_.get(), in.data_size_ * sizeof(dtype));
    data_size_ = in.data_size_;
    format_ = in.format_;
    return *this;
 }

  Tensor(const Tensor& in) {
    batch_ = in.batch_;
    w_ = in.w_;
    h_ = in.h_;
    c_ = in.c_;
    data_ = std::move(in.data_);
    data_size_ = in.data_size_;
    format_ = in.format_;
  }
  
  Tensor(Tensor&& in) {
    batch_ = in.batch_;
    w_ = in.w_;
    h_ = in.h_;
    c_ = in.c_;
    data_ = std::move(in.data_);
    data_size_ = in.data_size_;
    format_ = in.format_;
  }

  Tensor(int n, int h, int w, int c, TensorFormat format,
         std::shared_ptr<T[]> buf)
      : batch_(n), h_(h), w_(w), c_(c) {
    data_size_ = n * h * w * c;
    data_ = buf;
    format_ = format;
  }

  Tensor(int n, int h, int w, int c, TensorFormat format = NCHW)
      : batch_(n), h_(h), w_(w), c_(c) {
    data_size_ = n * h * w * c;
    data_ = std::shared_ptr<T[]>(new T[data_size_]);
    std::fill(data_.get(),data_.get()+data_size_,0);
    format_ = format;
  }

  Tensor clone() {
    Tensor out(batch_, h_, w_, c_, format_);
    out.data_ = std::shared_ptr<dtype[]>(new dtype[data_size_]);
    std::fill(out.data_.get(),out.data_.get()+data_size_,0);
    memcpy(out.data_.get(), data_.get(), data_size_ * sizeof(dtype));
    return out;
  }

  void swap(Tensor& in) {
    assert(format_ == in.format_);
    std::swap(batch_, in.batch_);
    std::swap(c_, in.c_);
    std::swap(h_, in.h_);
    std::swap(w_, in.w_);

    std::swap(data_size_, in.data_size_);
    data_.swap(in.data_);
  }

  void toNCHW() {
    if (format_ == NCHW) {
      return;
    }

    std::shared_ptr<T[]> temp(new T[data_size_]);
    std::fill(temp.get(),temp.get()+data_size_,0);
    for (int b = 0; b < batch_; b++) {
      for (int c = 0; c < c_; c++) {
        for (int i = 0; i < h_; i++) {
          for (int j = 0; j < w_; j++) {
            atNCHW(b, i, j, c, temp.get()) = atNHWC(b, i, j, c, data_.get());
          }
        }
      }
    }
    data_ = std::move(temp);
    format_ = NCHW;
  }

  void toNHWC() {
    if (format_ == NHWC) {
      return;
    }
    std::shared_ptr<T[]> temp(new T[data_size_]);
    std::fill(temp.get(),temp.get()+data_size_,0);
    for (int b = 0; b < batch_; b++) {
      for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
          for (int c = 0; c < c_; c++) {
            atNHWC(b, i, j, c, temp.get()) = atNCHW(b, i, j, c, data_.get());
          }
        }
      }
    }
    data_ = std::move(temp);
    format_ = NHWC;
  }

  T& atNCHW(int n, int h, int w, int c, T* array) {
    return array[c * h_ * w_ + h * w_ + w + n * h_ * w_ * c_];
  }

  T& atNHWC(int n, int h, int w, int c, T* array) {
    return array[h * w_ * c_ + w * c_ + c + n * h_ * w_ * c_];
  }

  T& at(int n, int h, int w, int c) {
    if (format_ == NCHW) {
      return atNCHW(n, h, w, c, data_.get());
    } else if (format_ == NHWC) {
      return atNHWC(n, h, w, c, data_.get());
    } else {
      throw std::runtime_error("unknown tensor format.");
    }
  }

  void print() {
    for (int b = 0; b < batch_; b++) {
      std::cerr << "{";
      for (int c = 0; c < c_; c++) {
        if (c % 3 == 0) std::cerr << TERMINAL_COLOR_RED;
        if (c % 3 == 1) std::cerr << TERMINAL_COLOR_YELLOW;
        if (c % 3 == 2) std::cerr << TERMINAL_COLOR_BLUE;
        std::cerr << "[";
        for (int h = 0; h < h_; h++) {
          std::cerr << "[";
          for (int w = 0; w < w_; w++) {
            std::cerr << std::setw(15) << at(b, h, w, c)
                      << (w == (w_ - 1) ? "" : ",");
          }
          std::cerr << "]" << (h == (h_ - 1) ? "" : ",\n");
        }
        std::cerr << "]" << (c == (c_ - 1) ? "" : ",\n");
        std::cerr << TERMINAL_COLOR_RESET;
      }
      std::cerr << "}" << std::endl;
    }
  }

  using dtype = T;

 public:
  int batch_;
  int h_;
  int w_;
  int c_;
  std::shared_ptr<T[]> data_;
  TensorFormat format_;
  uint64_t data_size_;
};

#endif