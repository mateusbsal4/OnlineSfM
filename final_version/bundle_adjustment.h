#ifndef BUNDLE_ADJUSTMENT_H
#define BUNDLE_ADJUSTMENT_H

#include "pipeline.h"

struct BAReprojectionError{
    BAReprojectionError(float observed_x, float observed_y);
    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const;
    static ceres::CostFunction* Create(const double observed_x,const double observed_y);    
    double observed_x;
    double observed_y;
    double fx = 4826.28455;
    double cx = 1611.73703;
    double fy = 4827.31363;
    double cy = 1330.23261;
};

#endif  //BUNDLE_ADJUSTMENT_H