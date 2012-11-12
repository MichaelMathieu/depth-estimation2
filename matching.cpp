#include<cmath>
#include<iostream>
#include<cassert>
#include<opencv/cv.h>
extern "C" {
#include<luaT.h>
#include<TH/TH.h>
}
using namespace std;

typedef THFloatTensor Tensor;
#define ID_TENSOR_STRING "torch.FloatTensor"
#define Tensor_(a) THFloatTensor_##a
typedef float Real;
typedef double accreal;
typedef unsigned char byte;
typedef unsigned short uint16;

#define TWO_BITS_PER_FILTER

#ifdef __ARM__
#define __NEON__
#endif
//#define _FILTER32
#ifndef __ARM__
#define _FILTER32
#endif

static int Binarize(lua_State *L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idlong = "torch.LongTensor";
  Tensor*        input = (Tensor      *)luaT_checkudata(L, 1, idreal);
  THLongTensor* output = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  Real       threshold = lua_tonumber(L, 3);

  const int N = input->size[0];
  const int h = input->size[1];
  const int w = input->size[2];
  const Real* ip = Tensor_(data)(input);
  long* op = THLongTensor_data(output);
  const long* const is = input->stride;
  const long* const os = output->stride;
  cout << N << " " << sizeof(long) << " " << output->size[2] << endl;

  int longSize = sizeof(long)*8;
  long* const op0 = op;
  const Real* iendh, *iendw, *const ip0 = ip;
  int shift, k;
//#pragma omp parallel for private(k, iendh, iendw, shift, ip, op)
  for (k = 0; k < N; ++k) {
#ifdef TWO_BITS_PER_FILTER
    op = op0 + (2*k/longSize)*os[2];
    shift = (2*k) % longSize;
#else
    op = op0 + (k/longSize)*os[2];
    shift = k % longSize;
#endif
    ip = ip0 + k*is[0];
    iendh = ip + h*is[1];
    while (ip != iendh) {
      iendw = ip + w*is[2];
      while (ip != iendw) {
	*op |= (((*ip) > threshold) << shift);
#ifdef TWO_BITS_PER_FILTER
	*op |= (((*ip) < -threshold) << (shift+1));
#endif
	ip += is[2];
	op += os[1];
      }
      ip += is[1] - w*is[2];
      op += os[0] - w*os[1];
    }
  }

  return 0;
}

static int BinaryMatching(lua_State *L) {
  const char* idlong = "torch.LongTensor";
  const char* idbyte = "torch.ByteTensor";
  THLongTensor* input1      = (THLongTensor*)luaT_checkudata(L, 1, idlong);
  THLongTensor* input2      = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THByteTensor* output      = (THByteTensor*)luaT_checkudata(L, 3, idbyte);
  THLongTensor* outputscore = (THLongTensor*)luaT_checkudata(L, 4, idlong);
  int           hmax        = lua_tointeger(L, 5);
  int           wmax        = lua_tointeger(L, 6);

  const int K = input1->size[2];
  const int h = input1->size[0];
  const int w = input1->size[1];
  const long* i1p = THLongTensor_data(input1);
  const long* i2p = THLongTensor_data(input2);
  byte* op  = THByteTensor_data(output);
  long* osp = THLongTensor_data(outputscore);
  const long* const i1s = input1->stride;
  const long* const i2s = input2->stride;
  const long* const os  = output->stride;
  const long* const oss = outputscore->stride;

  //printf("%d, %d, %d\n", i1s[0], i1s[1], i1s[2]);
  //printf("%d, %d, %d\n", i2s[0], i2s[1], i2s[2]);

#ifdef _FILTER32
  int bestsum, sum;
  int bestdx = 0, bestdy = 0;
  int x, y, dx, dy, k;
  int dxmin=0, dxmax=wmax, dymin=0, dymax=hmax;
#else
  int bestsum[2], sum[2];
  int bestdx[2], bestdy[2];
  int x, y, dx, dy, k;
  int dxmin=0, dxmax=wmax, dymin=0, dymax=hmax;
#endif
#ifdef __NEON__

//#pragma omp parallel for private(y, x, dy, dx, sum, k, bestsum) firstprivate(bestdx, bestdy,dxmin,dxmax,dymin,dymax)
  for (y = 0; y < h; ++y) {
    if (y < 3*h/5) dymax = 3*hmax/5; else dymax = hmax;
    if (y > 2*h/5) dymin = 2*hmax/5; else dymin = 0;
#ifdef _FILTER32
    for (x = 0; x < w; ++x) {
      if (x < 3*w/5) dxmax = 3*wmax/5; else dxmax = wmax;
      if (x > 2*w/5) dxmin = 2*wmax/5; else dxmin = 0;
      bestsum = 127;

      for (dy = dymin; dy < dymax; ++dy){
        char max_array[16];
        int *argptr[3];
        argptr[0] = (int *)(i1p + (y*i1s[0]+x*i1s[1]) );
        argptr[1] = (int *)(i2p + ((y+dy)*i2s[0]+x*i2s[1]) );
        argptr[2] = (int *)max_array;
        //argptr[3] = &dxmax;
	__asm__ __volatile__ (
          "ldr         r0, [%0]         @ Load src ptr \n\t"
          "ldr         r1, [%0, #4]     @ Load src2 ptr \n\t"
          "ldr         r2, [%0, #8]     @ Load dst ptr \n\t"
          "@ldr         r3, [%0, #12]    @ Load dx \n\t"
          "vld1.32     d0, [r0]         @ Load src1 k0, k1\n\t"
          "@vld1.32     d1, [r0]         @ duplicate src1 k0, k1\n\t"
          "vmov.32     d1, d0           @ duplicate src1 k0, k1\n\t"
          "vld1.32     {d2-d3}, [r1]!   @ Load src2 dx[0,1]\n\t"
          "vld1.32     {d4-d5}, [r1]!   @ Load src2 dx[2,3]\n\t"
          "vld1.32     {d6-d7}, [r1]!   @ Load src2 dx[4,5]\n\t"
          "vld1.32     {d8-d9}, [r1]!   @ Load src2 dx[6,7]\n\t"
          "vld1.32     {d10-d11}, [r1]! @ Load src2 dx[8,9]\n\t"
          "vld1.32     {d12-d13}, [r1]! @ Load src2 dx[10,11]\n\t"
          "vld1.32     {d14-d15}, [r1]! @ Load src2 dx[12,13]\n\t"
          "vld1.32     {d16-d17}, [r1]! @ Load src2 dx[14,15]\n\t"
          "veor.32     q9, q0, q1       @ ExOR dx[0,1]\n\t"
          "veor.32     q10, q0, q2      @ ExOR dx[2,3]\n\t"
          "veor.32     q11, q0, q3      @ ExOR dx[4,5]\n\t"
          "veor.32     q12, q0, q4      @ ExOR dx[6,7]\n\t"
          "veor.32     q13, q0, q5      @ ExOR dx[8,9]\n\t"
          "veor.32     q14, q0, q6      @ ExOR dx[10,11]\n\t"
          "veor.32     q15, q0, q7      @ ExOR dx[12,13]\n\t"
          "vcnt.i8     q9, q9           @ cnt bit dx[0,1]\n\t"
          "vcnt.i8     q10, q10         @ cnt bit dx[2,3]\n\t"
          "vcnt.i8     q11, q11         @ cnt bit dx[4,5]\n\t"
          "vcnt.i8     q12, q12         @ cnt bit dx[6,7]\n\t"
          "vcnt.i8     q13, q13         @ cnt bit dx[8,9]\n\t"
          "vcnt.i8     q14, q14         @ cnt bit dx[10,11]\n\t"
          "vcnt.i8     q15, q15         @ cnt bit dx[12,13]\n\t"
          "vpadd.i8    d18, d18, d19    @ sum 8 bits dx[0,1]\n\t"
          "vpadd.i8    d20, d20, d21    @ sum 8 bits dx[2,3]\n\t"
          "vpadd.i8    d22, d22, d23    @ sum 8 bits dx[4,5]\n\t"
          "vpadd.i8    d24, d24, d25    @ sum 8 bits dx[6,7]\n\t"
          "vpadd.i8    d26, d26, d27    @ sum 8 bits dx[8,9]\n\t"
          "vpadd.i8    d28, d28, d29    @ sum 8 bits dx[10,11]\n\t"
          "vpadd.i8    d30, d30, d31    @ sum 8 bits dx[12,13]\n\t"
          "vpadd.i8    d18, d18, d20    @ sum 8 bits dx[0,3]\n\t"
          "veor.32     q10, q0, q8      @ ExOR dx[14,15]\n\t"
          "vpadd.i8    d22, d22, d24    @ sum 8 bits dx[4,7]\n\t"
          "vcnt.i8     q10, q10         @ cnt bit dx[14,15]\n\t"
          "vpadd.i8    d26, d26, d28    @ sum 8 bits dx[8,11]\n\t"
          "vpadd.i8    d20, d20, d21    @ sum 8 bits dx[14,15]\n\t"
          "vpadd.i8    d30, d30, d20    @ sum 8 bits dx[12,15]\n\t"
          "vpadd.i8    d18, d18, d22    @ sum 8 bits dx[0,7]\n\t"
          "vpadd.i8    d19, d26, d30    @ sum 8 bits dx[8,15]\n\t"
          "vpadd.i8    d19, d26, d30    @ sum 8 bits dx[8,15]\n\t"
          "vst1.8      {d18-d19}, [r2]  @ Store previous elements \n\t"
          :
          :"r" (argptr)
          : "cc", "r0", "r1", "r2", "memory",
            "q0", "q1", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15",
            "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
            "d12", "d13", "d14", "d15","d16", "d17", "d18", "d19",
            "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27","d28", "d29", "d30", "d31"
          );
        // printf("max array: %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n",
        //        max_array[0], max_array[1], max_array[2], max_array[3],
        //        max_array[4], max_array[5], max_array[6], max_array[7],
        //        max_array[8], max_array[9], max_array[10], max_array[11],
        //        max_array[12], max_array[13], max_array[14], max_array[15]);

        for (dx = dxmin; dx < dxmax; ++dx) {
          if (max_array[dx] < bestsum) {
            bestsum = max_array[dx];
            bestdx = dx;
            bestdy = dy;
          }
        }
        // printf("bsumx: %d\tdx:%d\tdy:%d\n",bestsum, bestdx, bestdy);
      }
      op [      y*os [1]+x*os [2]] = bestdy;
      op [os[0]+y*os [1]+x*os [2]] = bestdx;
      osp[      y*oss[0]+x*oss[1]] = bestsum;
      // printf("bsum: %d\tdx:%d\tdy:%d\n",bestsum, bestdx, bestdy);
#else
      for (x = 0; x < w; x=x+2) {
        if (x < 3*w/5) dxmax = 3*wmax/5; else dxmax = wmax;
        if (x > 2*w/5) dxmin = 2*wmax/5; else dxmin = 0;
        bestsum[0] = 127;
        bestsum[1] = 127;
        for (dy = dymin; dy < dymax; ++dy){
          char max_array[2][16];
          int *argptr[4];
          argptr[0] = (int *)(i1p + (y*i1s[0]+x*i1s[1]) );
          argptr[1] = (int *)(i2p + ((y+dy)*i2s[0]+x*i2s[1]) );
          argptr[2] = (int *)&max_array[0];
          argptr[3] = (int *)&max_array[1];
          //argptr[3] = &dxmax;
          __asm__ __volatile__ (
            "ldr         r0, [%0]         @ Load src ptr \n\t"
            "ldr         r1, [%0, #4]     @ Load src2 ptr \n\t"
            "ldr         r2, [%0, #8]     @ Load dst ptr px1\n\t"
            "ldr         r3, [%0, #12]    @ Load dst ptr px2\n\t"
            "vld1.32     d0[0], [r0]!        @ Load src1 px0\n\t"
            "vld1.32     d2[0], [r0]      @ Load src1 px1\n\t"
            "vdup.32     q0, d0[0]        @ duplicate px0 in Q0\n\t"
            "vld1.32     {d4-d5}, [r1]!   @ Load src2 dx[0,3]\n\t"
            "vld1.32     {d6-d7}, [r1]!   @ Load src2 dx[4,7]\n\t"
            "vdup.32     q1, d2[0]        @ duplicate px1 in Q1\n\t"
            "vld1.32     {d8-d9}, [r1]!   @ Load src2 dx[8,11]\n\t"
            "vld1.32     {d10-d11}, [r1]! @ Load src2 dx[12,15]\n\t"
            "veor.32     q6, q1, q2       @ ExOR px1 dx[0,3]\n\t"
            "veor.32     q7, q1, q3       @ ExOR px1 dx[4,7]\n\t"
            "veor.32     q8, q1, q4       @ ExOR px1 dx[8,11]\n\t"
            "veor.32     q9, q1, q5       @ ExOR px1 dx[12,15]\n\t"
            "veor.32     q2, q0, q2       @ ExOR px0 dx[0,3]\n\t"
            "veor.32     q3, q0, q3       @ ExOR px0 dx[4,7]\n\t"
            "veor.32     q4, q0, q4       @ ExOR px0 dx[8,11]\n\t"
            "veor.32     q5, q0, q5       @ ExOR px0 dx[12,15]\n\t"
            "vcnt.i8     q6, q6           @ cnt bit px1 dx[0,3]  \n\t"
            "vcnt.i8     q7, q7           @ cnt bit px1 dx[4,7]  \n\t"
            "vcnt.i8     q8, q8           @ cnt bit px1 dx[8,11] \n\t"
            "vcnt.i8     q9, q9           @ cnt bit px1 dx[12,15]\n\t"
            "vcnt.i8     q2, q2           @ cnt bit px0 dx[0,3]\n\t"
            "vcnt.i8     q3, q3           @ cnt bit px0 dx[4,7]\n\t"
            "vcnt.i8     q4, q4           @ cnt bit px0 dx[8,11]\n\t"
            "vcnt.i8     q5, q5           @ cnt bit px0 dx[12,15]\n\t"
            "vpadd.i8    d12, d12, d13    @ sum 8 bits px1 dx[0,3]  \n\t"
            "vpadd.i8    d14, d14, d15    @ sum 8 bits px1 dx[4,7]  \n\t"
            "vpadd.i8    d16, d16, d17    @ sum 8 bits px1 dx[8,11] \n\t"
            "vpadd.i8    d18, d18, d19    @ sum 8 bits px1 dx[12,15]\n\t"
            "vpadd.i8    d4,  d4,  d5     @ sum 8 bits px0 dx[0,3]  \n\t"
            "vpadd.i8    d6,  d6,  d7     @ sum 8 bits px0 dx[4,7]  \n\t"
            "vpadd.i8    d8,  d8,  d9     @ sum 8 bits px0 dx[8,11] \n\t"
            "vpadd.i8    d10, d10, d11    @ sum 8 bits px0 dx[12,15]\n\t"
            "vpadd.i8    d12, d12, d14    @ sum 8 bits px1 dx[0,7]  \n\t"
            "vpadd.i8    d13, d16, d18    @ sum 8 bits px1 dx[8,15] \n\t"
            "vpadd.i8    d4,  d4,  d6     @ sum 8 bits px0 dx[0,7]  \n\t"
            "vpadd.i8    d5,  d8,  d10    @ sum 8 bits px0 dx[8,15] \n\t"
            "vst1.8      {d12-d13}, [r3]  @ Store result px1 \n\t"
            "vst1.8      {d4-d5},   [r2]  @ Store result px0 \n\t"
            "@vst1.8      {d2-d3}, [r3]  @ Store result px1 \n\t"
            "@vst1.8      {d0-d1},   [r2]  @ Store result px0 \n\t"
            :
            :"r" (argptr)
            : "cc", "r0", "r1", "r2", "r3", "memory",
              "q0", "q1", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15",
              "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15","d16", "d17", "d18", "d19",
              "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27","d28", "d29", "d30", "d31"
            );
          // printf("max array: %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n",
          //        max_array[0][0], max_array[0][1], max_array[0][2], max_array[0][3],
          //        max_array[0][4], max_array[0][5], max_array[0][6], max_array[0][7],
          //        max_array[0][8], max_array[0][9], max_array[0][10], max_array[0][11],
          //        max_array[0][12], max_array[0][13], max_array[0][14], max_array[0][15]);
          // printf("max array: %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n",
          //        max_array[1][0], max_array[1][1], max_array[1][2], max_array[1][3],
          //        max_array[1][4], max_array[1][5], max_array[1][6], max_array[1][7],
          //        max_array[1][8], max_array[1][9], max_array[1][10], max_array[1][11],
          //        max_array[1][12], max_array[1][13], max_array[1][14], max_array[1][15]);

          for (dx = dxmin; dx < dxmax; ++dx) {
            if (max_array[0][dx] < bestsum[0]) {
              bestsum[0] = max_array[0][dx];
              bestdx[0] = dx;
              bestdy[0] = dy;
            }
            if (max_array[1][dx] < bestsum[1]) {
              bestsum[1] = max_array[1][dx];
              bestdx[1] = dx;
              bestdy[1] = dy;
            }
          }

          // printf("bsumx: %d\tdx:%d\tdy:%d\n",bestsum[0], bestdx[0], bestdy[0]);
        }
        op [      y*os [1]+x*os [2]] = bestdy[0];
        op [os[0]+y*os [1]+x*os [2]] = bestdx[0];
        osp[      y*oss[0]+x*oss[1]] = bestsum[0];
        op [      y*os [1]+(x+1)*os [2]] = bestdy[1];
        op [os[0]+y*os [1]+(x+1)*os [2]] = bestdx[1];
        osp[      y*oss[0]+(x+1)*oss[1]] = bestsum[1];
        // printf("bsum: %d\tdx:%d\tdy:%d\n",bestsum, bestdx, bestdy);
#endif
      }
    }

#else // __NEON__

//#pragma omp parallel for private(y, x, dy, dx, sum, k, bestsum) firstprivate(bestdx, bestdy, dxmin, dxmax, dymin, dymax)
  for (y = 0; y < h; ++y) {
    if (y < 3*h/5) dymax = 3*hmax/5; else dymax = hmax;
    if (y > 2*h/5) dymin = 2*hmax/5; else dymin = 0;
    for (x = 0; x < w; ++x) {
      if (x < 3*w/5) dxmax = 3*wmax/5; else dxmax = wmax;
      if (x > 2*w/5) dxmin = 2*wmax/5; else dxmin = 0;
      bestsum = 2000000000;
      for (dy = dymin; dy < dymax; ++dy)
	for (dx = dxmin; dx < dxmax; ++dx) {
	  sum = 0;
	  for (k = 0; k < K; ++k)
	    sum += __builtin_popcountl(i1p[y*i1s[0]+x*i1s[1]+k*i1s[2]] ^
				       i2p[(y+dy)*i2s[0]+(x+dx)*i2s[1]+k*i2s[2]]);
	  if (sum < bestsum) {
	    bestsum = sum;
	    bestdx = dx;
	    bestdy = dy;
	  }
	}
      op [      y*os [1]+x*os [2]] = bestdy;
      op [os[0]+y*os [1]+x*os [2]] = bestdx;
      osp[      y*oss[0]+x*oss[1]] = bestsum;
    }
  }

#endif // __NEON__

  return 0;
}

static int MedianFilter(lua_State *L) {
  const char* idbyte = "torch.ByteTensor";
  THByteTensor* input = (THByteTensor*)luaT_checkudata(L, 1, idbyte);
  int k = lua_tointeger(L, 2);

  const int h = input->size[1];
  const int w = input->size[2];
  byte* const ip = THByteTensor_data(input);
  const long* const is = input->stride;

  cv::Mat_<uint16> input_cv(h, w);
  uint16* pcv = (uint16*)input_cv.data;
  int cvstep = input_cv.step1();
  byte *ip1 = ip, *ip2 = ip + is[0], *ipendh = ip + h*is[1], *ipendw;
  while (ip1 != ipendh) {
    ipendw = ip1 + w*is[2];
    while (ip1 != ipendw) {
      *pcv = (*ip1)*256 + (*ip2);
      ++pcv; ip1 += is[2]; ip2 += is[2];
    }
    ip1 += is[1] - w*is[2];
    ip2 += is[1] - w*is[2];
    pcv += cvstep - w;
  }
  cv::medianBlur(input_cv, input_cv, k);
  pcv = (uint16*)input_cv.data;
  ip1 = ip; ip2 = ip + is[0]; ipendh = ip + h*is[1];
  while (ip1 != ipendh) {
    ipendw = ip1 + w*is[2];
    while (ip1 != ipendw) {
      *ip1 = (*pcv) / 256;
      *ip2 = (*pcv) - (*ip1)*256;
      ++pcv; ip1 += is[2]; ip2 += is[2];
    }
    ip1 += is[1] - w*is[2];
    ip2 += is[1] - w*is[2];
    pcv += cvstep - w;
  }

  return 0;
}

static int Merge(lua_State *L) {
  const char* idbyte = "torch.ByteTensor";
  const char* idlong = "torch.LongTensor";
  THByteTensor* input1  = (THByteTensor*)luaT_checkudata(L, 1, idbyte);
  THLongTensor* input1s = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THByteTensor* input2  = (THByteTensor*)luaT_checkudata(L, 3, idbyte);
  THLongTensor* input2s = (THLongTensor*)luaT_checkudata(L, 4, idlong);
  THByteTensor* output  = (THByteTensor*)luaT_checkudata(L, 5, idbyte);
  byte          hhwin   = lua_tointeger(L, 6);
  byte          hwwin   = lua_tointeger(L, 7);

  const int h = input1->size[1];
  const int w = input1->size[2];
  const byte* i1p  = THByteTensor_data(input1);
  const long* i1sp = THLongTensor_data(input1s);
  const byte* i2p  = THByteTensor_data(input2);
  const long* i2sp = THLongTensor_data(input2s);
  byte* op   = THByteTensor_data(output);
  const long* const i1s  = input1 ->stride;
  const long* const i1ss = input1s->stride;
  const long* const i2s  = input2 ->stride;
  const long* const i2ss = input2s->stride;
  const long* const os   = output ->stride;

#if 0
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j) {
      if (i1sp[i*i1ss[0]+j*i1ss[1]]<=i2sp[(i/2)*i2ss[0]+(j/2)*i2ss[1]]) {
	op[      i*os[1]+j*os[2]] = i1p[       i*i1s[1]+j*i1s[2]]+hhwin;
	op[os[0]+i*os[1]+j*os[2]] = i1p[i1s[0]+i*i1s[1]+j*i1s[2]]+hwwin;
      } else {
	op[      i*os[1]+j*os[2]] = i2p[       (i/2)*i2s[1]+(j/2)*i2s[2]]*2;
	op[os[0]+i*os[1]+j*os[2]] = i2p[i2s[0]+(i/2)*i2s[1]+(j/2)*i2s[2]]*2;
      }
    }
#else
  const byte *const i1p0 = i1p, *const i2p0 = i2p;
  byte *const op0 = op;
  const long* const i1sp0 = i1sp, *const i2sp0 = i2sp, *i1spend;
  int c, i;
  const int wincr = (w/2)*2*i1ss[1];
#ifdef __ARM__
//#pragma omp parallel for private(i1sp, i1spend, i2sp, i1p, i2p, op, c)
#endif
  for (i = 0; i < h; ++i) {
    i1sp = i1sp0 + i*i1ss[0];
    i1spend = i1sp + wincr;
    i2sp = i2sp0 + (i/2)*i2ss[0];
    i1p = i1p0 + i*i1s[1];
    i2p = i2p0 + (i/2)*i2s[1];
    op = op0 + i*os[1];
    while (i1sp != i1spend) {
      c = ((*i1sp) <= (*i2sp));
      *op         = c*(*i1p          + hhwin) + (!c)*(*i2p         )*2;
      *(op+os[0]) = c*(*(i1p+i1s[0]) + hwwin) + (!c)*(*(i2p+i2s[0]))*2;
      i1sp += i1ss[1];
      i1p += i1s[2];
      op += os[2];
      c = ((*i1sp) <= (*i2sp));
      *op         = c*(*i1p          + hhwin) + (!c)*(*i2p         )*2;
      *(op+os[0]) = c*(*(i1p+i1s[0]) + hwwin) + (!c)*(*(i2p+i2s[0]))*2;
      i1sp += i1ss[1]; i2sp += i2ss[1];
      i1p += i1s[2]; i2p += i2s[2];
      op += os[2];
    }
  }
#endif

  return 0;
}

static int SizeofLong(lua_State *L) {
  lua_pushinteger(L, sizeof(long));
  return 1;
}
static int UseNeon(lua_State *L) {
#ifdef __NEON__
  lua_pushboolean(L, 1);
#else
  lua_pushboolean(L, 0);
#endif
  return 0;
}

static const struct luaL_reg libmatching[] = {
  {"binarize", Binarize},
  {"binaryMatching", BinaryMatching},
  {"medianFilter", MedianFilter},
  {"merge", Merge},
  {"sizeofLong", SizeofLong},
  {"useNeon", UseNeon},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_libmatching (lua_State *L) {
  luaL_openlib(L, "libmatching", libmatching, 0);
  return 1;
}
