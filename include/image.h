#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string_view>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

enum ImageFormat : uint8_t {
  BGR_F32,
  BGR_U8,
  GRAY_F32,
  GRAY_U8,
};

class Image {
 public:
  Image(int w, int h, int c) : w_(w), h_(h), c_(c) {
    data_.reset(new float[w_*h_*c_]);
  }

  float& at(int x, int y, int c) {
    assert(x < w_ && y < h_ && c < c_);
    return data_[c * h_ * w_ + y * w_ + x];
  }

  float get(int x, int y, int c) {
    assert(x < w_ && y < h_ && c < c_);
    return data_[c * h_ * w_ + y * w_ + x];
  }
  void set(int x, int y, int c, float val) {
    if (x < 0 || y < 0 || c < 0 || x >= w_ || y >= h_ || c >= c_) return;
    assert(x < w_ && y < h_ && c < c_);
    data_[c * h_ * w_ + y * w_ + x] = val;
  }
  void add(int x, int y, int c, float val) {
    assert(x < w_ && y < h_ && c < c_);
    data_[c * h_ * w_ + y * w_ + x] += val;
  }
  int w_;
  int h_;
  int c_;
  std::shared_ptr<float[]> data_;
};

Image resize_image(Image im, int w, int h)
{
    Image resized(w, h, im.c_);   
    Image part(w, im.h_, im.c_);
    int r, c, k;
    float w_scale = (float)(im.w_- 1) / (w - 1);
    float h_scale = (float)(im.h_ - 1) / (h - 1);
    for(k = 0; k < im.c_; ++k){
        for(r = 0; r < im.h_; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w_ == 1){
                  val = im.get(im.w_ - 1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * im.at(ix, r, k) + dx * im.at(ix+1, r, k);
                }
                part.set(c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c_; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * part.at( c, iy, k);
                resized.at( c, r, k)=val;
            }
            if(r == h-1 || im.h_ == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * part.get( c, iy+1, k);
                resized.at(c, r, k)+=val;
            }
        }
    }

    return resized;
}

void rgbgr_image(Image im) {
    int i;
    for (i = 0; i < im.w_ * im.h_; ++i) {
        float swap = im.data_[i];
        im.data_[i] = im.data_[i + im.w_ * im.h_ * 2];
        im.data_[i + im.w_ * im.h_ * 2] = swap;
    }
}
Image load_image_stb(const char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);

    if(channels) c = channels;
    int i,j,k;
    Image im(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data_[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    free(data);
    return im;
}
Image load_image_cv(cv::Mat m, int channels) {

    Image im(m.cols, m.rows, m.channels());

    int h = m.rows;
    int w = m.cols;
    int c = m.channels();

    for (int k = 0; k < c; ++k) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                im.data_[k * w * h + i * w + j] =
                    m.at<cv::Vec3b>(i, j)[k] / 255.0;
            }
        }
    }
    rgbgr_image(im);
    return im;
}

void fill_image(Image m, float s) {
    int i;
    for (i = 0; i < m.h_ * m.w_ * m.c_; ++i) m.data_[i] = s;
}
void embed_image(Image source, Image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c_; ++k){
        for(y = 0; y < source.h_; ++y){
            for(x = 0; x < source.w_; ++x){
                float val = source.at(x,y,k);
                dest.at(dx+x, dy+y, k) = val;
            }
        }
    }
}
Image letterbox_image(Image im, int w, int h)
{
    int new_w = im.w_;
    int new_h = im.h_;
    if (((float)w/im.w_) < ((float)h/im.h_)) {
        new_w = w;
        new_h = (im.h_ * w)/im.w_;
    } else {
        new_h = h;
        new_w = (im.w_ * h)/im.h_;
    }
    Image resized = resize_image(im, new_w, new_h);
    Image boxed(w, h, im.c_);
    fill_image(boxed, .5);
    //int i;
    //for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2); 
    return boxed;
}


#endif