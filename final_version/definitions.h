#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define CERES_FOUND true
#define EIGEN_DONT_VECTORIZE

#include <iostream>
#include <algorithm>
#include <string>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/viz.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <assert.h>
#include <ctype.h>
#include <tuple>
#include <cmath>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xset_operation.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <typeinfo>
#include <algorithm>
#include <limits>
#include <iterator>
#include <fstream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

using namespace std;
using namespace cv;
using namespace cv::utils::logging;
using namespace xt;

#endif  //DEFINITIONS_H