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
const char* idreal = "torch.FloatTensor";
const char* idfloat = "torch.FloatTensor";
const char* idlong = "torch.LongTensor";
#define Tensor_(a) THFloatTensor_##a
typedef float Real;
typedef double accreal;
typedef unsigned char byte;
typedef unsigned short uint16;
typedef cv::Mat_<float> matf;

static int IntegralImage(lua_State *L) {
  THFloatTensor* input = (THFloatTensor*)luaT_checkudata(L, 1, idfloat);
  THFloatTensor* output = (THFloatTensor*)luaT_checkudata(L, 2, idfloat);

  const int h = input->size[0];
  const int w = input->size[1];
  float* ip = THFloatTensor_data(input);
  float* op = THFloatTensor_data(output);
  const long* const is = input->stride;
  const long* const os = output->stride;
  
  const matf input_cv(h, w, ip, is[0]*sizeof(float));
  matf output_cv(h+1, w+1, op, os[0]*sizeof(float));
  cv::integral(input_cv, output_cv, CV_32F);

  return 0;
}

inline float iimageSum(const float* iimage, const long* const is,
		       int x1, int y1, int x2, int y2) {
  return iimage[is[0]*y1+is[1]*x1]+iimage[is[0]*y2+is[1]*x2]
    - iimage[is[0]*y1+is[1]*x2] - iimage[is[0]*y2+is[1]*x1];
}

static int FilterImage(lua_State *L) {
  THFloatTensor* iimage  = (THFloatTensor*)luaT_checkudata(L, 1, idfloat);
  THLongTensor * filters = (THLongTensor *)luaT_checkudata(L, 2, idlong );
  THLongTensor * output  = (THLongTensor *)luaT_checkudata(L, 3, idlong );
  int hmax = lua_tointeger(L, 4);
  int wmax = lua_tointeger(L, 5);

  const int h = iimage->size[0]-1;
  const int w = iimage->size[1]-1;
  const int N = filters->size[0];
  const float* ip = THFloatTensor_data(iimage );
  const long*  fp = THLongTensor_data (filters);
  long*        op = THLongTensor_data (output );
  const long* const is = iimage->stride;
  const long* const fs = filters->stride;
  const long* const os = output->stride;

  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < h-hmax; ++i)
    for (int j = 0; j < w-wmax; ++j)
      for (int k = 0; k < N; ++k) {
	const int bit = iimageSum(ip, is, j+fp[fs[0]*k+fs[1]],i+fp[fs[0]*k],
				  j+fp[fs[0]*k+3*fs[1]],i+fp[fs[0]*k+2*fs[1]])
	  > iimageSum(ip, is, j+fp[fs[0]*k+5*fs[1]],i+fp[fs[0]*k+4*fs[1]],
		      j+fp[fs[0]*k+7*fs[1]],i+fp[fs[0]*k+6*fs[1]]);
	  op[i*os[0]+j*os[1]] |= bit << k;
      }

  return 0;
}

static const struct luaL_reg libfiltering[] = {
  {"integralImage", IntegralImage},
  {"filterImage", FilterImage},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_libfiltering (lua_State *L) {
  luaL_openlib(L, "libfiltering", libfiltering, 0);
  return 1;
}