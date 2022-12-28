#include <arm_neon.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr uint64_t GRAY8_LENGTH = 1280 * 800;
constexpr uint64_t GRAY16_LENGTH = GRAY8_LENGTH * 2;

bool GetRawImage(const std::string& file, uint8_t* gray16) {
  std::ifstream ifs(file, std::ios::binary);
  bool ret;
  if (ifs.is_open()) {
    long cnt = 0;
    // load octets from raw
    while (cnt < GRAY16_LENGTH) {
      ifs.read(reinterpret_cast<char*>(gray16 + cnt), sizeof(uint8_t));
      ++cnt;
    }
    ret = true;
  } else {
    std::cout << "open raw error" << std::endl;
    ret = false;
  }
  ifs.close();
  return ret;
}

void gray16togray8(uint8_t* src, uint8_t* dst, int width, int height) {
  uint32_t sum = 0;
  uint8_t* pix;
  pix = src;
  pix++;
  uint8_t* target;
  target = dst;
  int border = width*height*0.75;
  for (int i = 0; i < width * height; i++) {
    *target = *pix;
    if (i < border) {
      if ((*target) > 200) {
        sum++;
      }
    }
    pix++;
    pix++;
    target++;
  }

  std::cout << "sum: " << sum << std::endl;
}

#ifdef __ARM_NEON__
void gray16togray8_neon(uint8_t* src, uint8_t* dst, int width, int height) {
  div_t src_32 = div(width * height * 2*0.75, 32);
  uint8x16_t thr_mask = vdupq_n_u8(200);  //大于200-255，小于200-0
  uint8x16_t one = vdupq_n_u8(1);
  uint32_t sum = 0;
  if (src_32.rem == 0) {
    while (src_32.quot-- > 0) {
      // std::cout<<"quot: "<<src_32.quot<<" rem: "<<src_32.rem <<", ";
      const uint8x16x2_t ab = vld2q_u8(src);
      vst1q_u8(dst, ab.val[1]);
      uint8_t test[16];
      uint8x16_t cmp = vcleq_u8(ab.val[1], thr_mask);
      uint8x16_t cmp_add_one = vaddq_u8(cmp, one);
      sum += vaddlvq_u8(cmp_add_one);
      src += 32;
      dst += 16;
    }

    std::cout << "sum: " << sum << std::endl;
    return;
  }
}
#endif

int main() {
  unsigned char t = -100;
  std::cout << "t: " << (int)t << std::endl;
  std::shared_ptr<uint8_t[]> gray8(new uint8_t[GRAY8_LENGTH]);
  std::shared_ptr<uint8_t[]> gray8_neon(new uint8_t[GRAY8_LENGTH]);
  std::shared_ptr<uint8_t[]> gray16(new uint8_t[GRAY16_LENGTH]);

  GetRawImage("1670398320013.raw", gray16.get());

  constexpr float N = 10.0;

  auto start = std::chrono::high_resolution_clock::now();

#if 1
  for (int i = 0; i < N; i++) {
    gray16togray8(gray16.get(), gray8.get(), 1280, 800);
  }
  auto end = std::chrono::high_resolution_clock::now();
#else
  for (int i = 0; i < N; i++) {
    gray16togray8_neon(gray16.get(), gray8_neon.get(), 1280, 800);
  }
  auto end = std::chrono::high_resolution_clock::now();
#endif

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "average time: " << duration / N << "ms" << std::endl;

  // std::ofstream ofs1("out.raw", std::ios::out | std::ios::binary);
  // ofs1.write(reinterpret_cast<char*>(gray8.get()), GRAY8_LENGTH);
  // std::ofstream ofs2("out_neon.raw", std::ios::out | std::ios::binary);
  // ofs2.write(reinterpret_cast<char*>(gray8_neon.get()), GRAY8_LENGTH);

  return 0;
}