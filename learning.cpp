#include<iostream>
#include<string>
#include<vector>
#include<bitset>
#include<utility>
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<opencv/cv.h>
#include<opencv/highgui.h>
using namespace std;
using namespace cv;

typedef Mat_<float> matf;
typedef Mat_<unsigned char> matb;
typedef Mat_<Vec3b> mat3b;

const int MAX_BITS = 1024*16;
typedef bitset<MAX_BITS> Match;

matf loadImage(const string & path) {
  Mat im0 = imread(path);
  matf output;
  if (im0.type() == CV_8UC3) {
    matb im0b;
    cvtColor(im0, im0b, CV_RGB2GRAY);
    im0b.convertTo(output, CV_32F);
  } else {
    im0.convertTo(output, CV_32F);
  }
  return output/255.0f;
}

void displayImage(const matf & im) {
  namedWindow("win");
  imshow("win", im);
  cvWaitKey(0);
}

struct Filter {
  int x1p, y1p, x2p, y2p;
  int x1m, y1m, x2m, y2m;
  Filter(int x1p, int y1p, int x2p, int y2p,
	 int x1m, int y1m, int x2m, int y2m)
    :x1p(x1p), y1p(y1p), x2p(x2p), y2p(y2p),
     x1m(x1m), y1m(y1m), x2m(x2m), y2m(y2m) {};
  string toString() const {
    string output = "";
    char buffer[32];
    for (int i = 0; i < 8; ++i) {
      sprintf(buffer, "%s%d", (i==0)?"":", ", (&x1p)[i]);
      output = output + buffer;
    }
    return output;
  }
};

inline float iimageSum(const matf & iimage, int xoff, int yoff,
		       const int* coords) {
  return iimage(yoff+coords[1],xoff+coords[0])
    + iimage(yoff+coords[3],xoff+coords[2])
    - iimage(yoff+coords[1],xoff+coords[2])
    - iimage(yoff+coords[3],xoff+coords[0]);
}

Match filterPixel(const matf & iimage, int x, int y,
		  const vector<Filter> & filters) {
  Match output;
  for (int i = 0; i < filters.size(); ++i) {
    output[i] = iimageSum(iimage, x, y, &(filters[i].x1p))
      < iimageSum(iimage, x, y, &(filters[i].x1m));
  }
  return output;
}

vector<Match> filterImage(const matf & im, const vector<Filter> & filters) {
  int hmax = 0, wmax = 0;
  for (int i = 0; i < filters.size(); ++i) {
    hmax = max(max(hmax, filters[i].y2p-filters[i].y1p),
	       filters[i].y2m-filters[i].y1m);
    wmax = max(max(wmax, filters[i].x2p-filters[i].x1p),
	       filters[i].x2m-filters[i].x1m);
  }
  matf iim;
  integral(im, iim, CV_32F);
  int h = im.size().height;
  int w = im.size().width;
  vector<Match> output((h-hmax)*(w-wmax));
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < h-hmax; ++i) {
    for (int j = 0; j < w-wmax; ++j) {
      output[i*(w-wmax)+j] = filterPixel(iim, j, i, filters);
    }
  }
  
  return output;
}

vector<Filter> generateFilters(const vector<int> & X, const vector<int> & Y) {
  vector<Filter> output;
  for (int i1 = 0; i1 < X.size(); ++i1)
    for (int j1 = 0; j1 < Y.size(); ++j1)
      for (int i2 = i1+1; i2 < X.size(); ++i2)
	for (int j2 = j1+1; j2 < Y.size(); ++j2)
	  for (int i1m = 0; i1m < X.size(); ++i1m)
	    for (int j1m = 0; j1m < Y.size(); ++j1m)
	      for (int i2m = i1m+1; i2m < X.size(); ++i2m)
		for (int j2m = j1m+1; j2m < Y.size(); ++j2m)
		  output.push_back(Filter(X[i1],Y[j1],X[i2],Y[j2],
					  X[i1m],Y[j1m],X[i2m],Y[j2m]));
  return output;
}

inline void filteredToMat(const vector<Match> & matchs, int i,
			  vector<float> & out) {
  for (int j = 0; j < matchs.size(); ++j)
    out[j] = (float)matchs[j][i];
}
double correlation(const vector<vector<float> > & selected,
		   const vector<float> & current) {
  double output = 0;
  for (int j = 0; j < selected.size(); ++j) {
    output = max(output, compareHist(current, selected[j], CV_COMP_CORREL));
  }
  return output;
}

vector<int> selectFilters(const vector<Match> & matchs, int n, float max_corr) {
  vector<pair<float, int> > idx(MAX_BITS);
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < MAX_BITS; ++i) {
    long double mean=0;
    for (int j = 0; j < matchs.size(); ++j)
      mean += matchs[j][i];
    mean = abs(mean/(long double)matchs.size() - 0.5);
    idx[i] = pair<float, int>(mean, i);
  }
  cout << "mean computed" << endl;
  sort(idx.begin(), idx.end());
  cout << "sorted" << endl;

  vector<vector<float> > selectedFloat;
  vector<float> currentFloat(matchs.size());
  vector<int> output;
  i = 0;
  while ((i < idx.size()) && (output.size() < n)) {
    filteredToMat(matchs, idx[i].second, currentFloat);
    double corr = correlation(selectedFloat, currentFloat);
    if (corr < max_corr) {
      //cout << output.size() << "/" << n << endl;
      selectedFloat.push_back(currentFloat);
      output.push_back(idx[i].second);
    }
    ++i;
  }
  
  return output;
}

void displayFilters(const vector<Filter> & filters) {
  int n = 20, n1=16;
  matf output(n*8,n*8,1.f);
  int oy = 0, ox = 0;
  for (int i = 0; i < filters.size(); ++i) {
    for (int j = 0; j < n1; ++j)
      for (int k = 0; k < n1; ++k)
	output(oy+k,ox+j) = 0.33;
    const Filter & f = filters[i];
    for (int j = f.x1p; j < f.x2p; ++j)
      for (int k = f.y1p; k < f.y2p; ++k)
	output(oy+k,ox+j) = 0.67;
    for (int j = f.x1m; j < f.x2m; ++j)
      for (int k = f.y1m; k < f.y2m; ++k)
	output(oy+k,ox+j) = 0.;
    ox += n;
    if (ox == n*8) {
      oy += n;
      ox = 0;
    }
  }
  displayImage(output);
}

int main (int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " imagename[s]" << endl;
    return -1;
  }
  vector<int> X, Y;
  X.push_back(0);
  X.push_back(4);
  X.push_back(8);
  X.push_back(12);
  X.push_back(16);
  Y.push_back(0);
  Y.push_back(4);
  Y.push_back(8);
  Y.push_back(12);
  Y.push_back(16);
  vector<Filter> filters = generateFilters(X, Y);
  cout << filters.size() << " possible filters" << endl;

  vector<Match> pixels;
  for (int i = 1; i < argc; ++i) {
    matf im0 = loadImage(argv[i]);
    matf im;
    resize(im0, im, Size(180, 240));
    vector<Match> pixels1 = filterImage(im, filters);
    pixels.insert(pixels.end(), pixels1.begin(), pixels1.end());
    cout << "Image " << argv[i] << " filtered" << endl;
  }

  vector<int> selected = selectFilters(pixels, 32, 0.7);
  
  cout << "{" << endl;
  for (int i = 0; i < selected.size(); ++i) {
    int t = selected[i];
    cout << "{" << filters[t].toString() << "}," << endl;
  }
  cout << "}" << endl;

  vector<Filter> selectedFilters;
  for (int i = 0; i < selected.size(); ++i)
    selectedFilters.push_back(filters[selected[i]]);
  displayFilters(selectedFilters);
  
  return 0;
}
