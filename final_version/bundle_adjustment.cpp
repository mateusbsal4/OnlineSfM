#include "definitions.h"
#include "pipeline.h"
#include "bundle_adjustment.h"

BAReprojectionError::BAReprojectionError(float observed_x, float observed_y): observed_x(observed_x), observed_y(observed_y){

} 

template <typename T>
bool BAReprojectionError::operator()(const T* const camera, const T* const point, T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];   
    ceres::AngleAxisRotatePoint(camera, point, p);
    assert(p[2] != 0.0);
    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];
    // Compute the center of distortion. 
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];
    // Compute final projected point position.
    T predicted_x = xp*fx+ cx;
    T predicted_y = yp*fy +cy;     
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
}

ceres::CostFunction* BAReprojectionError::Create(const double observed_x, const double observed_y) {
    return (new ceres::AutoDiffCostFunction<BAReprojectionError, 2, 6, 3>(new BAReprojectionError(observed_x, observed_y)));
}

tuple<vector<vector<double>>, vector<Point3f>, vector<Point2f>, vector<int>, vector<int>> StructureFromMotion::prepare_optimization_input(vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud, vector<vector<Point2f>> tracks, vector<vector<int>> masks){
    Mat r,R,T;
    vector<vector<double>> camera_params;
    for(size_t k = Rs.size() - ba_window; k<Rs.size(); k++){
        tie(R, T) = invert_reference_frame(Rs[k], Ts[k]);
        Rodrigues(R, r);        //rotation vec - angles of rot
        vector<double> rTvec;  //rotation and translation vectors concatenated horizontally in one array
        rTvec.push_back(r.at<double>(0,0));
        rTvec.push_back(r.at<double>(1,0));      //first 3 params - rotation
        rTvec.push_back(r.at<double>(2,0));                
        rTvec.push_back(T.at<double>(0,0));                
        rTvec.push_back(T.at<double>(1,0));      //last 3 - translation
        rTvec.push_back(T.at<double>(2,0));
        camera_params.push_back(rTvec);
    } 
    xarray<int> xcloud_mask = get_not_nan_index_mask(cloud);
    vector<int> cloud_mask(xcloud_mask.begin(), xcloud_mask.end());
    xarray<float> cloud_reindex = {0.0};   
    cloud_reindex.resize({cloud.size()});
    cloud_reindex = all_nan_list(cloud_reindex);
    float q = 0;
    for(size_t p = 0; p < cloud_reindex.size(); p++){
        if(std::find(xcloud_mask.begin(), xcloud_mask.end(), p) != xcloud_mask.end()){
            cloud_reindex(p) = q++;
        }
    }
    q = 0;
    vector<int> camera_indices, point_indices;
    vector<Point2f> points_2d;
    for(size_t p = tracks.size() - ba_window; p<tracks.size(); p++){
        vector<int> track_mask = masks[p];
        xarray<int> xtrack_mask = adapt(track_mask, {track_mask.size()});
        xarray<int> intersection_mask = get_intersection_mask(cloud_mask, track_mask);  
        xarray<int> track_bool_mask = isin(xtrack_mask, intersection_mask);
        vector<int> camera_indices_row = full_of_ints(intersection_mask.size(), q++);
        camera_indices.insert(camera_indices.end(), camera_indices_row.begin(), camera_indices_row.end());
        vector<int> point_indices_row;
        for(size_t c = 0; c<cloud_reindex.size(); c++){
            if(std::find(intersection_mask.begin(), intersection_mask.end(), c) != intersection_mask.end()){
                point_indices_row.push_back((int)cloud_reindex(c));
            }
        }  
        point_indices.insert(point_indices.end(), point_indices_row.begin(), point_indices_row.end());
        vector<Point2f> track = tracks[p];
        xarray<Point2f> xtrack = adapt(track, {track.size()});
        xarray<Point2f> xpoints_2d_row = filter(xtrack, track_bool_mask);
        vector<Point2f> points_2d_row(xpoints_2d_row.begin(), xpoints_2d_row.end());
        points_2d.insert(points_2d.end(), points_2d_row.begin(), points_2d_row.end());
        assert(camera_indices_row.size() == point_indices_row.size());                
        assert(point_indices_row.size() == points_2d_row.size()); 
    }
    assert(camera_indices.size() == point_indices.size());
    assert(point_indices.size() == points_2d.size());
    xarray<Point3f> xcloud = adapt(cloud, {cloud.size()});
    vector<Point3f> points_3d;
    for(size_t c = 0; c<xcloud.size(); c++){                                                  
        if(std::find(xcloud_mask.begin(), xcloud_mask.end(), c) != xcloud_mask.end()){
            points_3d.push_back(xcloud(c));
        }
    }
   return make_tuple(camera_params, points_3d, points_2d, camera_indices, point_indices);
}

void StructureFromMotion::run_ba(vector<Mat>& Rs, vector<Mat>& Ts, vector<Point3f>& cloud, vector<vector<Point2f>>& tracks, vector<vector<int>>& masks){
    vector<vector<double>> camera_params;
    vector<Point3f> points_3d;
    vector<Point2f> points_2d;
    vector<int> camera_indices, point_indices;
    tie(camera_params, points_3d, points_2d, camera_indices, point_indices) = prepare_optimization_input(Rs, Ts, cloud, tracks, masks);
    double optimization_params[points_2d.size()][9];
    ceres::Problem problem;
    for(size_t p = 0; p< points_2d.size(); p++){
        Point3f point_3d = points_3d[point_indices[p]];
        vector<double> rTvecs = camera_params[camera_indices[p]];
        optimization_params[p][0] = rTvecs[0];
        optimization_params[p][1] = rTvecs[1];         //first 3 - rotation vec
        optimization_params[p][2] = rTvecs[2];
        optimization_params[p][3] = rTvecs[3];                
        optimization_params[p][4] = rTvecs[4];         //following 3 - translation vec
        optimization_params[p][5] = rTvecs[5];                
        optimization_params[p][6] = (double)point_3d.x;
        optimization_params[p][7] = (double)point_3d.y;                    //last 3 - point coordinates
        optimization_params[p][8] = (double)point_3d.z;
        ceres::CostFunction* cost_function = BAReprojectionError::Create(
        (double)points_2d[p].x, (double)points_2d[p].y);
        problem.AddResidualBlock(cost_function,
            nullptr /* squared loss */,
            &optimization_params[p][0],    //this holds a pointer to the first element of the rtvector
            &optimization_params[p][6]);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    for(size_t p = 0; p< points_2d.size(); p++){        //converting back to pipeline structure
        camera_params[camera_indices[p]][0] = optimization_params[p][0];                
        camera_params[camera_indices[p]][1] = optimization_params[p][1];
        camera_params[camera_indices[p]][2] = optimization_params[p][2];
        camera_params[camera_indices[p]][3] = optimization_params[p][3];
        camera_params[camera_indices[p]][4] = optimization_params[p][4];
        camera_params[camera_indices[p]][5] = optimization_params[p][5];
        points_3d[point_indices[p]].x = (float)optimization_params[p][6];                 
        points_3d[point_indices[p]].y = (float)optimization_params[p][7]; 
        points_3d[point_indices[p]].z = (float)optimization_params[p][8]; 
    }
    size_t q = 0;
    for(size_t k = Rs.size() - ba_window; k< Rs.size(); k++){        //modifying Rs & Ts
        Mat r = Mat::zeros(3, 1, CV_64F);;                
        Mat T = Mat::zeros(3, 1, CV_64F);;  
        Mat R;  
        r.at<double>(0,0) = camera_params[q][0];
        r.at<double>(1,0) = camera_params[q][1];      //first 3 params - rotation
        r.at<double>(2,0) = camera_params[q][2];                
        T.at<double>(0,0) = camera_params[q][3];                
        T.at<double>(1,0) = camera_params[q][4];      //last 3 - translation
        T.at<double>(2,0) = camera_params[q++][5];
        Rodrigues(r, R);
        tie(Rs[k], Ts[k]) = invert_reference_frame(R, T);
    }
    assert(q == camera_params.size());
    xarray<bool> nan_bool_mask = get_nan_bool_mask(cloud);
    q = 0;
    for(size_t k = 0; k< cloud.size(); k++){        //modifying cloud_3d
        if(!nan_bool_mask(k)){
            cloud[k] = points_3d[q++];
        }
    }
    assert(q == points_3d.size());
}