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
  
  const int N = input->size[0];
  const int h = input->size[1];
  const int w = input->size[2];
  const Real* ip = Tensor_(data)(input);
  long* op = THLongTensor_data(output);
  const long* const is = input->stride;
  const long* const os = output->stride;

  int longSize = sizeof(long)*8;
#if 0
  int iInt, shift, i, j, k;
#pragma omp parallel for private(iInt, shift, i, j)
  for (k = 0; k < N; ++k) {
    iInt  = k / longSize;
    shift = k % longSize;
    for (i = 0; i < h; ++i)
      for (j = 0; j < w; ++j) {
	op[i*os[0]+j*os[1]+iInt*os[2]] |= ((ip[k*is[0]+i*is[1]+j*is[2]] > threshold) << shift);
      }
  }
#else
  long* const op0 = op;
  const Real* iendh, *iendw, *const ip0 = ip;
  int shift, k;
#pragma omp parallel for private(iendh, iendw, shift, ip, op)
  for (k = 0; k < N; ++k) {
    op = op0 + (k/longSize)*os[2];
    ip = ip0 + k*is[0];
    iendh = ip + h*is[1];
    shift = k % longSize;
    while (ip != iendh) {
      iendw = ip + w*is[2];
      while (ip != iendw) {
	*op |= (((*ip) > threshold) << shift);
	ip += is[2];
	op += os[1];
      }
      ip += is[1] - w*is[2];
      op += os[0] - w*os[1];
    }
    //op += os[0] - h*os[1];
  }
#endif

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
  const long* i1s = input1->stride;
  const long* i2s = input2->stride;
  const long* os  = output->stride;
  const long* oss = outputscore->stride;

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
	  for (k = 0; k < 1; ++k)
	    sum += __builtin_popcountl(i1p[y*i1s[0]+x*i1s[1]+k*i1s[2]] ^
				       i2p[(y+dy)*i2s[0]+(x+dx)*i2s[1]+k*i2s[2]]);
	  //cout << sum << " " << bestsum << endl;
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

  return 0;
}

static int MedianFilter(lua_State *L) {
  const char* idbyte = "torch.ByteTensor";
  THByteTensor* input = (THByteTensor*)luaT_checkudata(L, 1, idbyte);
  int k = lua_tointeger(L, 2);

  const int h = input->size[1];
  const int w = input->size[2];
  byte* ip = THByteTensor_data(input);
  const long* is = input->stride;

  cv::Mat_<unsigned short> input_cv(h, w);
  int i, j, t;
  for (i = 0; i < h; ++i)
    for (j = 0; j < w; ++j)
      input_cv(i, j) = ip[i*is[1]+j*is[2]]*256+ip[is[0]+i*is[1]+j*is[2]];
  cv::medianBlur(input_cv, input_cv, k);
  for (i = 0; i < h; ++i)
    for (j = 0; j < w; ++j) {
      t = input_cv(i, j)/256;
      ip[     +i*is[1]+j*is[2]] = t;
      ip[is[0]+i*is[1]+j*is[2]] = input_cv(i, j)-t*256;
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
  const long* i1s  = input1 ->stride;
  const long* i1ss = input1s->stride;
  const long* i2s  = input2 ->stride;
  const long* i2ss = input2s->stride;
  const long* os   = output ->stride;
  
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j) {      
      if (i1sp[i*i1ss[0]+j*i1ss[1]]<=i2sp[(i/2)*i2ss[0]+(j/2)*i2ss[1]]) {
	op[      i*os[1]+j*os[2]] = i1p[       i*i1s[1]+j*i1s[2]]+hhwin;
	op[os[0]+i*os[1]+j*os[2]] = i1p[i1s[0]+i*i1s[1]+j*i1s[2]]+hwwin;
	//op[      i*os[1]+j*os[2]] = 100;
	//op[os[0]+i*os[1]+j*os[2]] = 100;
      } else {
	op[      i*os[1]+j*os[2]] = i2p[       (i/2)*i2s[1]+(j/2)*i2s[2]]*2;
	op[os[0]+i*os[1]+j*os[2]] = i2p[i2s[0]+(i/2)*i2s[1]+(j/2)*i2s[2]]*2;
      }
    }
     
  return 0;
}

static const struct luaL_reg libmatching[] = {
  {"binarize", Binarize},
  {"binaryMatching", BinaryMatching},
  {"medianFilter", MedianFilter},
  {"merge", Merge},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_libmatching (lua_State *L) {
  luaL_openlib(L, "libmatching", libmatching, 0);
  return 1;
}
