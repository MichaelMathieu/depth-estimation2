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

static int Binarize(lua_State *L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idlong = "torch.LongTensor";
  Tensor*        input = (Tensor      *)luaT_checkudata(L, 1, idreal);
  THLongTensor* output = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  Real       threshold = lua_tonumber(L, 3);
  
  int N = input->size[0];
  int h = input->size[1];
  int w = input->size[2];
  Real* ip = Tensor_(data)(input);
  long* op = THLongTensor_data(output);
  long* is = input->stride;
  long* os = output->stride;

  int longSize = sizeof(long)*8;
  int iInt, shift;
  for (int k = 0; k < N; ++k) {
    iInt  = k / longSize;
    shift = k % longSize;
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j) {
	op[i*os[0]+j*os[1]+iInt*os[2]] |= ((ip[k*is[0]+i*is[1]+j*is[2]] > threshold) << shift);
      }
  }

  return 0;
}

static int BinaryMatching(lua_State *L) {
  const char* idlong = "torch.LongTensor";
  const char* idbyte = "torch.ByteTensor";
  THLongTensor* input1 = (THLongTensor*)luaT_checkudata(L, 1, idlong);
  THLongTensor* input2 = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THByteTensor* output = (THByteTensor*)luaT_checkudata(L, 3, idbyte);
  int         hmax   = lua_tointeger(L, 4);
  int         wmax   = lua_tointeger(L, 5);
  
  int K = input1->size[2];
  int h = input1->size[0];
  int w = input1->size[1];
  long* i1p = THLongTensor_data(input1);
  long* i2p = THLongTensor_data(input2);
  byte* op  = THByteTensor_data(output);
  long* i1s = input1->stride;
  long* i2s = input2->stride;
  long* os  = output->stride;

  unsigned int bestsum, sum;
  int bestdx = 0, bestdy = 0;
  int x, y, dx, dy, k;
#pragma omp parallel for private(x, dy, dx, sum, k, bestsum, bestdx, bestdy)
  for (y = 0; y < h; ++y)
    for (x = 0; x < w; ++x) {
      bestsum = -1;
      for (dy = 0; dy < hmax; ++dy)
	for (dx = 0; dx < wmax; ++dx) {
	  sum = 0;
	  for (k = 0; k < K; ++k)
	    sum += __builtin_popcountl(i1p[y*i1s[0]+x*i1s[1]+k*i1s[2]] ^
				       i2p[(y+dy)*i2s[0]+(x+dx)*i2s[1]+k*i2s[2]]);
	  //cout << sum << " " << bestsum << endl;
	  if (sum < bestsum) {
	    bestsum = sum;
	    bestdx = dx;
	    bestdy = dy;
	  }
	}
      op[      y*os[1]+x*os[2]] = bestdy;
      op[os[0]+y*os[1]+x*os[2]] = bestdx;
    }

  return 0;
}

static int MedianFilter(lua_State *L) {
  const char* idbyte = "torch.ByteTensor";
  THByteTensor* input = (THByteTensor*)luaT_checkudata(L, 1, idbyte);
  int k = lua_tointeger(L, 2);

  int h = input->size[1];
  int w = input->size[2];
  byte* ip = THByteTensor_data(input);
  long* is = input->stride;

  cv::Mat_<unsigned short> input_cv(h, w);
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j)
      input_cv(i, j) = ip[i*is[1]+j*is[2]]*256+ip[is[0]+i*is[1]+j*is[2]];
  cv::medianBlur(input_cv, input_cv, k);
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j) {
      int t = input_cv(i, j)/256;
      ip[     +i*is[1]+j*is[2]] = t;
      ip[is[0]+i*is[1]+j*is[2]] = input_cv(i, j)-t*256;
    }
  
  return 0;
} 

static const struct luaL_reg libmatching[] = {
  {"binarize", Binarize},
  {"binaryMatching", BinaryMatching},
  {"medianFilter", MedianFilter},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_libmatching (lua_State *L) {
  luaL_openlib(L, "libmatching", libmatching, 0);
  return 1;
}
