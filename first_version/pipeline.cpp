#include <iostream>
#include <algorithm>
#include <string>
#include <numeric>
#define CERES_FOUND true
#define EIGEN_DONT_VECTORIZE
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

//THESE CRITERIA ARE SENSITIVE TO CHANGES
//TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,50,0.01);
TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,50,0.01);
Size subPixWinSize(10,10), winSize(31,31);            //example (standard OpenCV) criteria

//TermCriteria termcrit(3,30,0.003);
//Size subPixWinSize(10,10), winSize(15,15);           // tcc criteria

int initial_indexes_shape = 1;   
xarray<int> indexes_2d({initial_indexes_shape});


string video_name;




struct BAReprojectionError {
  BAReprojectionError(float observed_x, float observed_y)
      : observed_x(observed_x), observed_y(observed_y) {} 

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
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

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<BAReprojectionError, 2, 6, 3>(
        new BAReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
  double fx = 4826.28455;
  double cx = 1611.73703;
  double fy = 4827.31363;
  double cy = 1330.23261;
};









class StructureFromMotion{
    private:
        vector<vector<Point2f>> tracks;
        vector<vector<int>> masks;
        vector<Point2f> track, features, prev_points, points, new_points;
        Mat frame, color, image, gray, prevGray;
        vector<int> frame_numbers, mask;
        //vector<xarray<double>> Rs, Ts; 
        vector<Mat> global_Rs, global_Ts;       
        vector<Point3f> cloud_3d = {};
        Matx33f K_viz = {4826.28455, 0.0, 1611.73703,
                     0.0, 4827.31363, 1330.23261,
                     0.0, 0.0, 1.0};

        Mat K = (Mat_<double>(3,3) << 4826.28455, 0, 1611.73703, 0, 4827.31363, 1330.23261, 0, 0, 1);
        const int skip_at_getter = 3; //skip_at_getter in tcc pipeline
        const int yield_period = 1;
        const LogTag TAG = LogTag("SfM", LOG_LEVEL_DEBUG);
        size_t frame_counter = 0;
        size_t i, j;
        size_t initial_counter = 0;
        //const int MAX_COUNT = 300;  //og was 300
        const int MAX_COUNT = 300; //for elef5
        //const int MAX_COUNT = 100;      //tcc param
        //const int MAX_COUNT = 98;
        const double closeness_threshold = 15;
        //const int min_features = 100; //og
        //const int min_features = 200; //for elef5 this should definitely be better 
        //const int min_features = 35; //tcc param
        //const int min_features = 49;
        const int min_features = 150; //for elef5
        bool needToInit = 1;
        //bool first_frame = 1;
        int start_index = 0;
        //float error_threshold = 50;     //empirically best value is ~1.5
        //float error_threshold = 1.5;      //note: this is not an absolute value, sometimes it is impossible to have so low a threshold for a given example
        float error_threshold = 8;  
        //float error_threshold = 4;
        //float error_threshold = 3.2;
        int init_reconstruction_frames = 5;
        int error_calculation_frames = 5;
        bool is_init = 1;
        double ransac_probability = 0.999999;
        double essential_mat_threshold = 5;
        double distance_thresh = 500;

        bool use_epnp = 1;
        //bool use_epnp = 0;
        //bool use_iterative_pnp = 0;
        bool use_iterative_pnp = 1;
        int min_number_of_points = 5;

        bool use_five_pt_algorithm = 0;
        //bool use_five_pt_algorithm = 1;
        bool use_solve_pnp = 1;
        int dropped_tracks = 0;
        
        int ba_window = 100;
        int adjust_path = 0;

        


    public:
        StructureFromMotion(){}


        int runSfM(VideoCapture cap){
            viz::Viz3d window("SfM Visualization");
            window.setWindowSize(Size(2000,2000));
            //window.setWindowPosition(Point(150,150));
            //window.setBackgroundColor(); // black by default
            vector<Affine3d> path;
            vector<Vec3f> point_cloud_est;
            for(;;){
                static int frame_number;
                frame_counter ++;
                if(frame_counter%(skip_at_getter+1) !=0){continue;} 
                if(check_end(cap)){break;}     //checks if video sequence has ended
                tie(frame_number, track, mask) = feature_detector();
                tracks.push_back(track);                                   
                masks.push_back(mask);
                frame_numbers.push_back(frame_number);
                if(tracks.size() < init_reconstruction_frames + error_calculation_frames){
                    continue;
                }
                if(is_init){
                    static float error;
                    vector<Point3f> init_cloud = {};
                    vector<Mat> init_Rs, init_Ts;
                    initial_counter = 0;
                    std::vector<vector<Point2f>> init_tracks = std::vector<vector<Point2f>> (tracks.begin() + dropped_tracks, tracks.begin()+init_reconstruction_frames + dropped_tracks);
                    std::vector<vector<int>> init_masks = std::vector<vector<int>> (masks.begin() + dropped_tracks, masks.begin()+init_reconstruction_frames + dropped_tracks);               
                    //std::vector<int> init_frame_numbers = std::vector<int> (frame_numbers.begin(), frame_numbers.begin()+init_reconstruction_frames);       
                    assert(init_tracks.size() == init_masks.size());                                                                                        
                    assert(init_tracks.size() == 5);
                    tie(init_Rs, init_Ts, init_cloud) = init_reconstruction(init_tracks, init_masks, init_Rs, init_Ts, init_cloud);
                    global_Rs = init_Rs;
                    global_Ts = init_Ts;
                    cloud_3d = init_cloud;
                    std::vector<vector<Point2f>> error_calc_tracks = std::vector<vector<Point2f>> (tracks.end() - error_calculation_frames, tracks.end());
                    std::vector<vector<int>> error_calc_masks = std::vector<vector<int>> (masks.end() - error_calculation_frames, masks.end());               
                    std::vector<int> error_calc_frame_numbers = std::vector<int> (frame_numbers.end() - error_calculation_frames, frame_numbers.end());
                    assert(error_calc_tracks.size() == 5);
                    assert(init_Rs.size() == 5);
                    if(!cloud_3d.empty()){
                        error = calculate_init_error(error_calc_tracks, error_calc_masks, cloud_3d);       
                    }
                    if(error > error_threshold){
                        dropped_tracks += 1;
                        //continue;
                    }
                    else{
                        is_init = 0;
                    }
                    //TO this line
                    continue;
                }
        
                vector<vector<Point2f>> remaining_tracks(tracks.end() - error_calculation_frames-1, tracks.end());
                vector<vector<int>> remaining_masks(masks.end() - error_calculation_frames-1, masks.end());                       
                vector<int> remaining_frame_numbers(frame_numbers.end()-error_calculation_frames-1, frame_numbers.end()); 
                
                tie(global_Rs, global_Ts, cloud_3d) = reconstruct(remaining_tracks, remaining_masks, global_Rs, global_Ts, cloud_3d); 
                if(global_Rs.size() % ba_window == 0){         //frame selected for BA
                    run_ba(global_Rs, global_Ts, cloud_3d, tracks, masks);      //reference to global objects is passed, so they are modified inside run_ba function
                    adjust_path = 1;
                }


                viz::WCloud cloud_widget(cloud_3d, viz::Color::white());
                //Affine3d viewer_pose(Vec3d(0, 0, -1000));
                //float scale_factor = 2.0; 
                //cv::Affine3d scale_transform = cv::viz::makeTransformToGlobal(Vec3f(0.0f,-2.0f,0.0f), Vec3f(-2.0f,0.0f,0.0f), Vec3f(0.0f,0.0f,-2.0f));
                //cloud_widget.setPose(scale_transform * cloud_widget.getPose());
                cloud_widget.setRenderingProperty(viz::POINT_SIZE, 3.0);
                window.showWidget("point_cloud", cloud_widget);
                if(global_Rs.size() > 0){  
                    if(adjust_path){        //this corrects the trajectory after each ba run   
                        for(size_t path_ctr = path.size() + 1 - ba_window; path_ctr < path.size(); path_ctr++){
                            path[path_ctr] = Affine3d(global_Rs[path_ctr], global_Ts[path_ctr]);
                        }
                        adjust_path = 0;
                    }
                    for(size_t path_counter = path.size(); path_counter<global_Rs.size(); path_counter ++){
                        path.push_back(Affine3d(global_Rs[path_counter], global_Ts[path_counter]));

                    }
                    window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::PATH, 0.1, viz::Color::blue()));
                    window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K_viz, 0.1, viz::Color::red()));
                    //window.setViewerPose(viewer_pose);                    
                }
                window.spinOnce(1, true);          
            }
            



            while(1){window.spinOnce(1, true);}     //keeps the window open after pipeline has ended
            return 0;
        }




        void run_ba(vector<Mat>& Rs, vector<Mat>& Ts, vector<Point3f>& cloud, vector<vector<Point2f>>& tracks, vector<vector<int>>& masks){
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



        tuple<vector<vector<double>>, vector<Point3f>, vector<Point2f>, vector<int>, vector<int>> prepare_optimization_input(vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud, vector<vector<Point2f>> tracks, vector<vector<int>> masks){
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




        vector<int> full_of_ints(size_t size, int data){
            vector<int> vec;
            for(size_t c = 0; c<size; c++){
                vec.push_back(data);
            }
            return vec;
        }

        tuple<vector<Mat>, vector<Mat>, vector<Point3f>> reconstruct(vector<vector<Point2f>> tracks, vector<vector<int>> masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud){
            Mat R, T; 
            vector<Point3f> new_pts;
            xarray<int> new_pt_indexes;
            tie(R, T, new_pts, new_pt_indexes) = calculate_projection(Rs[Rs.size() - 1], Ts[Ts.size() - 1], tracks, masks, cloud);
            if(!R.empty() && !T.empty()){
                if(!new_pts.empty()){
                    cloud = add_points_to_cloud(new_pts, new_pt_indexes, cloud);
                }
                Rs.push_back(R);
                Ts.push_back(T);
            }
            return make_tuple(Rs, Ts, cloud);
        }

        bool check_end(VideoCapture cap){
            cap >> frame;
            return (frame.empty())?1:0;
        }


        std::tuple<int, vector<Point2f>, vector<int>> feature_detector(){
            Mat image;
            color = frame;
            Mat mask = Mat::zeros(color.size(), color.type());
            Mat tracking_window;
            color.copyTo(image);
            cvtColor(image, gray, COLOR_BGR2GRAY);
            if( needToInit )
            {
                get_new_features(mask, image);
                needToInit = 0;
                start_index = amax(indexes_2d)()+1;
            }


            else if( !prevGray.empty() ){
                track_features(mask, image);
            }   


            add(image, mask, tracking_window);
            imshow("LK Tracker", tracking_window);
            waitKey(10);

            std::swap(points, prev_points);
            cv::swap(prevGray, gray);
            if(prev_points.size() < min_features){
                needToInit = 1;
            }
            if(frame_counter % yield_period ==0){
                vector<int> inds(indexes_2d.begin(), indexes_2d.end());       
                return  std::make_tuple(frame_counter, prev_points, inds);
            }

        }



        void get_new_features(Mat mask, Mat image){
            //goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);   //example (standard OpenCV) params - block size (3) is too small! This selects too many features
            //cornerSubPix(gray, new_points, subPixWinSize, Size(-1,-1), termcrit);
            //goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.01, 10, Mat(), 7, 3, 0, 0.04);   
            goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.01, 10, Mat(), 7, 3, 0, 0.04);   
            cornerSubPix(gray, new_points, subPixWinSize, Size(-1,-1), termcrit);
            //goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.5, 15, Mat(), 11,3, 0, 0.04);    //tcc params
            //cornerSubPix(gray, new_points, subPixWinSize, Size(-1,-1), termcrit);
            if(!points.empty()){
                track_features(mask, image);
                match_features();
            }
            else{
                points = new_points;
                indexes_2d.resize({points.size()});
                indexes_2d = arange(points.size());
            }
        }

        void track_features(Mat mask, Mat image){
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, prev_points, points, status, err, winSize,
                                 3, termcrit, 0, 0.0001);
            for( i = j = 0; i < points.size(); i++ )
            {
                if( !status[i] )
                    continue;
                points[j++] = points[i];
                line(mask, prev_points[i], points[i], Scalar(255,0,0), 2);
                circle( image, points[i], 3, Scalar(0,255,0), -1, 8);
            }
            points.resize(j);
            auto xstatus = adapt(status, {status.size()});
            indexes_2d = filter(indexes_2d, xstatus);
        }



        void match_features(){
            size_t n = points.size();
            size_t m = new_points.size();
            vector<size_t> shape = {n, m};
            xarray<int> closeness_table(shape);
            for( i = 0; i < n; i++ ){
                for( j = 0; j < m; j++){
                    if(norm(points[i],new_points[j]) <= closeness_threshold){
                        closeness_table(i,j)=1;
                    }
                    else{
                        closeness_table(i,j) =0;
                    }
                }
            }
            xarray<int> new_points_mask({m});
            new_points_mask = sum(closeness_table, 0);
            int ones_in_mask = 0;
            for( i = 0; i<m; i++ ){
                if(new_points_mask(i)==0){
                    new_points_mask(i) = 1;
                    ones_in_mask ++;
                }
                else{
                    new_points_mask(i) = 0;
                }
            }
            auto old_features  = adapt(points, {n});
            auto new_features = adapt(new_points, {m});



            new_features = filter(new_features, new_points_mask);
            assert(ones_in_mask == new_features.size());
            auto new_indexes = arange(0, ones_in_mask)+start_index;
            auto features = xt::hstack(xtuple(old_features, new_features));
            //auto features = xt::hstack(xtuple(old_features, new_features));
            indexes_2d = xt::hstack(xtuple(indexes_2d, new_indexes));

            vector<Point2f> feats(features.begin(), features.end());            

            points = feats;

            assert(points.size() ==indexes_2d.size());
        }



        double norm(Point2f &a, Point2f &b){
            return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
        }




        float calculate_init_error(vector<vector<Point2f>> error_calc_tracks, vector<vector<int>> error_calc_masks, vector<Point3f> pt_cloud){
            size_t c;
            vector<Mat> error_calc_Rs;
            vector<Mat> error_calc_Ts;
            Mat R, T, empty_R, empty_T;
            assert(error_calc_masks.size() == error_calc_tracks.size());
            for(c =0; c<error_calc_masks.size(); c++){
                tie(R, T) = solve_pnp(error_calc_tracks[c], error_calc_masks[c], empty_R, empty_T, pt_cloud);
                error_calc_Rs.push_back(R);
                error_calc_Ts.push_back(T);             
            }
            return calculate_reconstruction_error(error_calc_Rs, error_calc_Ts, error_calc_tracks, error_calc_masks, pt_cloud); //xadrez_cc memory leakage occurs here
        }

        float calculate_reconstruction_error(vector<Mat> ec_Rs, vector<Mat> ec_Ts, vector<vector<Point2f>> ec_tracks, vector<vector<int>> ec_masks, vector<Point3f> pt_cloud){
            size_t c;
            Mat R, T, R_cam, r_cam_vec, T_cam, distCoeffs;
            vector<Point2f> original_track, projected_track;
            vector<int> track_mask;
            xarray<int> xtrack_mask;
            assert(ec_Rs.size() == ec_tracks.size());
            xarray<int> xcloud_mask = get_not_nan_index_mask(pt_cloud);
            vector<int> cloud_mask(xcloud_mask.begin(), xcloud_mask.end());     


            vector<float> errors;
            for(c= 0 ; c< ec_tracks.size(); c++){
                vector<Point3f> filtered_cloud;
                R = ec_Rs[c];
                T = ec_Ts[c];
                original_track = ec_tracks[c];
                track_mask = ec_masks[c];
                xtrack_mask = adapt(track_mask, {track_mask.size()});    


                xarray<int> intersection_mask = get_intersection_mask(cloud_mask, track_mask);     
                auto track_bool_mask = isin(xtrack_mask, intersection_mask);
                tie(R_cam, T_cam) = invert_reference_frame(R, T);
                Rodrigues(R_cam, r_cam_vec);
                for(i = 0; i<pt_cloud.size(); i++){
                    if(std::find(intersection_mask.begin(), intersection_mask.end(), i) != intersection_mask.end()){
                        filtered_cloud.push_back(pt_cloud[i]);
                    }
                }   
                projectPoints(filtered_cloud, r_cam_vec, T_cam, K, distCoeffs, projected_track);    
                auto xprojected_pts = adapt(projected_track, {projected_track.size()}); 
                auto xoriginal_pts = adapt(original_track, {original_track.size()});    //all fine til here
                assert(intersection_mask.size() == xprojected_pts.size());
                assert(filter(xoriginal_pts, track_bool_mask).size() == xprojected_pts.size());
                auto xdelta = filter(xoriginal_pts, track_bool_mask) - xprojected_pts;
                assert(xdelta(0).x == filter(xoriginal_pts, track_bool_mask)(0).x - xprojected_pts(0).x);
                assert(xdelta(0).y == filter(xoriginal_pts, track_bool_mask)(0).y - xprojected_pts(0).y);
                vector<float> errors_per_frame;
                for(i = 0; i< xdelta.size(); i++){
                    errors_per_frame.push_back(sqrt((xdelta(i).x)*(xdelta(i).x)+(xdelta(i).y)*(xdelta(i).y)));
                }
                errors.push_back(my_mean(errors_per_frame));    
            }
            return my_mean(errors);
        }


        tuple<vector<Mat>, vector<Mat>, vector<Point3f>> init_reconstruction(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud){
            Mat R, T;
            vector<Point3f> pts_3d;
            xarray<int> indexes;
            for(initial_counter = 0; initial_counter < init_reconstruction_frames; initial_counter++){      
                vector<vector<Point2f>> init_tracks_sliced  = std::vector<vector<Point2f>> (init_tracks.begin(), init_tracks.begin()+initial_counter+1);                     
                vector<vector<int>> init_masks_sliced = std::vector<vector<int>> (init_masks.begin(), init_masks.begin()+initial_counter+1);                       
                if(cloud.empty()){
                    if(init_masks_sliced.size() > 1 && init_tracks_sliced[0].size() >= 5){
                        tie(Rs, Ts, cloud) = five_pt_init(init_tracks_sliced, init_masks_sliced, Rs, Ts, cloud);
                        assert(Rs.size() == 2);
                    }   
                    continue;
                }
                tie(R, T, pts_3d, indexes) = calculate_projection(Rs[Rs.size()-1], Ts[Ts.size()-1], init_tracks_sliced, init_masks_sliced, cloud);
                if(!pts_3d.empty()){
                    cloud = add_points_to_cloud(pts_3d, indexes, cloud); 
                }
                Rs.push_back(R);
                Ts.push_back(T);

            }
            return make_tuple(Rs, Ts, cloud);  
        }





        tuple<Mat, Mat, vector<Point3f>, xarray<int>> calculate_projection(Mat prev_R, Mat prev_T, vector<vector<Point2f>> tracks, vector<vector<int>> masks, vector<Point3f> cloud){
            int threshold = 40;
            assert((use_epnp == 1 && use_five_pt_algorithm ==0) || (use_epnp ==0 && use_five_pt_algorithm == 1 && use_iterative_pnp == 1));
            Mat R, T;
            vector<vector<Point2f>> track_pair;
            xarray<int> pair_mask;
            tie(track_pair, pair_mask)  = get_last_track_pair(tracks, masks);
            if(use_five_pt_algorithm){
                tie(R, T) = five_pt(track_pair, pair_mask, prev_R, prev_T);
            }
            if(use_solve_pnp){
                tie(R, T) = solve_pnp(tracks[tracks.size()-1], masks[masks.size() -1], R, T, cloud);       
            }
            if(!prev_T.empty()){
                if((abs(T.at<double>(0,0)) > abs(threshold*prev_T.at<double>(0,0)) || abs(T.at<double>(1,0)) > abs(threshold*prev_T.at<double>(1,0)) || abs(T.at<double>(2,0))> abs(threshold*prev_T.at<double>(2,0)))){
                    Mat empty_R, empty_T;
                    vector<Point3f> empty_points;
                    return make_tuple(empty_R, empty_T, empty_points, pair_mask);
                }
            }
            vector<Point3f> points_3d = triangulate(prev_R.t(), -(prev_R.t())*prev_T, R.t(), -(R.t())*T, track_pair);           //careful here!! Cv operations won't work with empty matrices
            return make_tuple(R, T, points_3d, pair_mask);  
        }

        
        tuple<Mat, Mat> solve_pnp(vector<Point2f> track, vector<int> mask, Mat R_est, Mat T_est, vector<Point3f> cloud){
            Mat R, T;
            if(use_epnp){
                tie(R, T) = solve_pnp_(track, mask, SOLVEPNP_EPNP, R_est, T_est, cloud);
                R_est = R;
                T_est = T;
            }   
            if(use_iterative_pnp){
                tie(R, T) = solve_pnp_(track, mask, SOLVEPNP_ITERATIVE, R_est, T_est, cloud);
            }
            return make_tuple(R, T);
        }

        tuple<Mat, Mat> solve_pnp_(vector<Point2f> track_slice, vector<int> track_mask, cv::SolvePnPMethod method, Mat R, Mat T, vector<Point3f> cloud){
           bool use_extrinsic_guess = (!R.empty() && !T.empty()) ? 1 : 0;
           xarray<int> xcloud_mask = get_not_nan_index_mask(cloud); 
           vector<int> cloud_mask(xcloud_mask.begin(), xcloud_mask.end());
           xarray<int> intersection_mask = get_intersection_mask(cloud_mask, track_mask);
           xarray<int> xtrack_mask = adapt(track_mask, {track_mask.size()});
           xarray<bool> track_bool_mask = isin(xtrack_mask, intersection_mask);
           if(intersection_mask.size()<min_number_of_points){
                return make_tuple(R,T);
           }
           tie(R, T) = invert_reference_frame(R,T);
           vector<Point3f> pts_cloud;
           for(i = 0; i<cloud.size(); i++){                                                  
            if(std::find(intersection_mask.begin(), intersection_mask.end(), i) != intersection_mask.end()){
                pts_cloud.push_back(cloud[i]);
            }
           }
           auto xtrack_slice = adapt(track_slice, {track_slice.size()});
           xtrack_slice = filter(xtrack_slice, track_bool_mask);
           vector<Point2f> img_points(xtrack_slice.begin(), xtrack_slice.end());
           Mat rvec, distCoeffs;
           if(!R.empty()){
                Rodrigues(R, rvec);
           }
           assert(pts_cloud.size() == img_points.size());
           solvePnP(pts_cloud, img_points, K, distCoeffs, rvec, T, use_extrinsic_guess, method);
           Rodrigues(rvec, R);
           tie(R, T) = invert_reference_frame(R, T);
           return make_tuple(R, T);
        }
    




        xarray<int> get_nan_index_mask(vector<Point3f> cloud){
            xarray<bool> nan_bool_mask = get_nan_bool_mask(cloud);
            return filter(arange(cloud.size()), nan_bool_mask);
        }

        xarray<int> get_not_nan_index_mask(vector<Point3f> cloud){
            xarray<bool> not_nan_bool_mask = !get_nan_bool_mask(cloud);
            return filter(arange(cloud.size()), not_nan_bool_mask);
        }



        xarray<bool> get_nan_bool_mask(vector<Point3f> cloud){
            vector<bool> nan_bool_mask;
            for(i = 0; i<cloud.size(); i++){
                if(isnan(cloud[i].x) || isnan(cloud[i].y) || isnan(cloud[i].z)){
                    nan_bool_mask.push_back(1);
                }
                else{
                    nan_bool_mask.push_back(0);
                }
            }
            auto xnan_bool_mask = adapt(nan_bool_mask, {nan_bool_mask.size()});
            return xnan_bool_mask;
        }


        tuple<vector<Mat>, vector<Mat>, vector<Point3f>> five_pt_init(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud){
            if(init_tracks.size()>2){
                init_tracks = vector<vector<Point2f>> (init_tracks.end()-1, init_tracks.end());
                init_masks = vector<vector<int>> (init_masks.end()-1, init_masks.end());
            }
            Mat R, T; 
            Rs = {Mat::eye(3,3,CV_64F)};
            Ts = {Mat::zeros(3,1,CV_64F)};

            vector<vector<Point2f>> track_pair;
            xarray<int> pair_mask;
            tie(track_pair, pair_mask)  = get_last_track_pair(init_tracks, init_masks);
            tie(R, T) = five_pt(track_pair, pair_mask, Rs[0], Ts[0]);
            if(R.empty()){
                vector<Mat> null_Rs, null_Ts;
                return make_tuple(null_Rs, null_Ts, cloud);
            }
            vector<Point3f> points3d;
            assert(track_pair[0].size() == pair_mask.size());
            assert(pair_mask.size() == amax(pair_mask)() + 1);
            if(pair_mask.size() != amax(pair_mask)() + 1){
                vector<Mat> Rs, Ts;                             //if init fails (treat the above assert)
                vector<Point3f> cloud;
                return make_tuple(Rs, Ts, cloud);
            }
            points3d = triangulate(Rs[0].t(), -(Rs[0].t())*Ts[0], R.t(), -(R.t())*T, track_pair);                                              //reconverting motion matrices to relative frame coordinate system is required by the OpenCV function
            assert(points3d.size() == pair_mask.size());
            cloud = points_to_cloud(points3d, pair_mask); 
            Rs.push_back(R);
            Ts.push_back(T);
            return make_tuple(Rs, Ts, cloud);
        }

        tuple<Mat, Mat> five_pt(vector<vector<Point2f>> track_pair, xarray<int> pair_mask, Mat prev_R, Mat prev_T){
            Mat E, five_pt_mask, mask, R, T; 
            E = findEssentialMat(track_pair[0], track_pair[1], K,  RANSAC, ransac_probability, essential_mat_threshold, five_pt_mask);       
            five_pt_mask.copyTo(mask);
            recoverPose(E, track_pair[0], track_pair[1], K, R, T, distance_thresh, five_pt_mask);

            tie(R, T) = invert_reference_frame(R,T);
            tie(R, T) = compose_rts(R, T, prev_R, prev_T);
            return make_tuple(R,T);
        }


        vector<Point3f> points_to_cloud(vector<Point3f> points3d, xarray<int> pair_mask){
            auto points_3d = adapt(points3d, {points3d.size()});
            xarray<Point3f> xcloud = {{0.0,0.0,0.0}};   
            xcloud.resize({amax(pair_mask)() +1});
            xcloud = all_nan(xcloud);
            xcloud = points_3d;
            assert(xcloud.size() == points_3d.size());
            assert(adapt(xcloud.shape()) == adapt(points_3d.shape()));            
            assert(xcloud.dimension() == points_3d.dimension());
            vector<Point3f> points_cloud(xcloud.begin(), xcloud.end()); //this WORKS

            return points_cloud;
        }

        vector<Point3f> add_points_to_cloud(vector<Point3f> points_3d, xarray<int> indexes, vector<Point3f> cloud){   //NOTE: this function only adds new points to the cloud, it does not replace any point that has already been triangulated.
            assert(points_3d.size() == indexes.size());                                                       
            assert(!cloud.empty());                                                                        
            xarray<int> cloud_mask = get_not_nan_index_mask(cloud);                                 
            xarray<int> new_points_mask = setdiff1d(indexes, cloud_mask);     
            xarray<Point3f> xcloud = adapt(cloud, {cloud.size()}); 
           
            if(amax(indexes)() > cloud.size() -1){
                xarray<Point3f> new_cloud = {{0.0,0.0,0.0}};   
                new_cloud.resize({2*amax(indexes)()});
                new_cloud = all_nan(new_cloud);
                new_cloud = double_filter(new_cloud, xcloud, cloud_mask);
                xcloud.resize({2*amax(indexes)()});
                xcloud = new_cloud;
            }   
            if(new_points_mask.size() >= 1){
                auto xpoints_3d = adapt(points_3d, {points_3d.size()});
                xarray<Point3f> new_points = filter(xpoints_3d, isin(indexes, new_points_mask));
                for(int i = 0; i< new_points.size(); i++){                
                    int index = new_points_mask(i);
                    xcloud(index) = new_points(i);
                }
            }  
            vector<Point3f> points_cloud(xcloud.begin(), xcloud.end());  
            return points_cloud;
        }

        xarray<Point3f> double_filter(xarray<Point3f> first_vec, xarray<Point3f> second_vec, xarray<int> mask1, xarray<int> mask2 = {}){
            for(const int & index: mask1){
                first_vec(index) = second_vec(index);
            }
            return first_vec;
        } 

        xarray<Point3f> all_nan(xarray<Point3f> vector_3d){
            for(i =0; i<vector_3d.size(); i++){
                vector_3d(i).x = numeric_limits<float>::quiet_NaN();            
                vector_3d(i).y = numeric_limits<float>::quiet_NaN();
                vector_3d(i).z = numeric_limits<float>::quiet_NaN();             
            }
            return vector_3d;
        }


        xarray<float> all_nan_list(xarray<float> list){
            for(i =0; i<list.size(); i++){
                list(i)= numeric_limits<float>::quiet_NaN();                    
            }
            return list;
        }

        vector<Point3f> triangulate(Mat R1, Mat T1, Mat R2, Mat T2, vector<vector<Point2f>> track_pair){
            vector<Point3f> points3d;
            if(R1.empty() || T1.empty() || R2.empty() || T2.empty()){
                return points3d;
            }
            xarray<double> P1({3,4});
            xarray<double> P2({3,4});
            xarray<double> xK = mat_to_xarray(K);
            xarray<double> xR1 = mat_to_xarray(R1);
            xarray<double> xR2 = mat_to_xarray(R2);
            xarray<double> xT1 = mat_to_xarray(T1);
            xarray<double> xT2 = mat_to_xarray(T2);
            P1 = linalg::dot(xK, hstack(xtuple(xR1, xT1)));
            P2 = linalg::dot(xK, hstack(xtuple(xR2, xT2)));
            Mat P_1, P_2, points4d;
            P_1 = xarray_to_mat_elementwise(P1);
            P_2 = xarray_to_mat_elementwise(P2);
            assert(track_pair[0].size() == track_pair[1].size());
            triangulatePoints(P_1, P_2, track_pair[0], track_pair[1], points4d);
            assert(points4d.cols == track_pair[0].size());
            assert(points4d.rows != 0 && points4d.cols !=0);
            points4d = points4d.t();
            Mat pts4d = points4d.reshape(4,1);
            convertPointsFromHomogeneous(pts4d, points3d);
            return points3d;
        }
        
        tuple<Mat, Mat> invert_reference_frame(Mat R, Mat T){
            if(R.empty()){
                return make_tuple(T, R);
            }
            return make_tuple(R.t(), -(R.t())*T);                                   //expressing motion matrices between current and previous frame
        }                                                                           // in global (first camera) coordinate system

        tuple<Mat, Mat> compose_rts(Mat R, Mat T, Mat prev_R, Mat prev_T){
            return make_tuple(prev_R*R, prev_T + prev_R*T);                      //this expresses motion from the current frame in respect to the first
        }
        
        cv::Mat xarray_to_mat_elementwise(xt::xarray<double> xarr)
        {
            int ndims = xarr.dimension();
            assert(ndims == 2);
            int nrows = xarr.shape()[0];
            int ncols = xarr.shape()[1];
            cv::Mat mat(nrows, ncols, CV_64FC1);
            for (int rr=0; rr<nrows; rr++)
            {
                for (int cc=0; cc<ncols; cc++)
                {
                    mat.at<double>(rr, cc) = xarr(rr, cc);
                }
            }
            return mat;
        }


        xt::xarray<double> mat_to_xarray(cv::Mat mat)
        {
            xt::xarray<double> res = xt::adapt((double*) mat.data, mat.cols * mat.rows, xt::no_ownership(), std::vector<std::size_t> {mat.rows, mat.cols});
            return res;
        }


        float my_mean(std::vector<float> const& v)
        {
            if(v.empty()){
                return 0;
            }
            float sum = std::accumulate(v.begin(), v.end(), 0.0f);
            return sum / v.size();
        }


        xarray<int> get_intersection_mask(vector<int> mask1, vector<int> mask2){
            vector<int> intersection_mask;
            for(i = 0; i<mask1.size(); i++){
                for(j=0; j<mask2.size(); j++){
                    if(mask1[i]==mask2[j]){          
                        intersection_mask.push_back(mask1[i]);
                    }
                }
            }
            sort(intersection_mask.begin(), intersection_mask.end());
            auto xintersection_mask = adapt(intersection_mask, {intersection_mask.size()});
            return xintersection_mask;
        }


        tuple<vector<vector<Point2f>>, xarray<int>> get_last_track_pair(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks){
            vector<Point2f> init_track1 = init_tracks[init_tracks.size()-1];
            vector<Point2f> init_track2 = init_tracks[init_tracks.size()-2];
            vector<int> init_mask1 = init_masks[init_masks.size()-1];
            vector<int> init_mask2 = init_masks[init_masks.size()-2];

            auto xpair_mask = get_intersection_mask(init_mask2, init_mask1);
            assert(get_intersection_mask(init_mask1, init_mask2) == get_intersection_mask(init_mask2, init_mask1));
            auto xmask1 = adapt(init_mask1, {init_mask1.size()});
            auto xmask2 = adapt(init_mask2, {init_mask2.size()});

            auto xtrack1 = adapt(init_track1, {init_track1.size()});
            auto xtrack2 = adapt(init_track2, {init_track2.size()});
            auto xtrack_pair0 = filter(xtrack1, isin(xmask1, xpair_mask));
            auto xtrack_pair1 = filter(xtrack2, isin(xmask2, xpair_mask));

            vector<Point2f> track_pair0(xtrack_pair0.begin(), xtrack_pair0.end());
            vector<Point2f> track_pair1(xtrack_pair1.begin(), xtrack_pair1.end());

            vector<vector<Point2f>> track_pair{track_pair1, track_pair0};


            return make_tuple(track_pair, xpair_mask);

        }



};









int main(int argc, char ** argv){
    StructureFromMotion sfm;
    if(argc==1){
        VideoCapture cap(0);
        return sfm.runSfM(cap);
    }
    else{
        string path_to_vid = "/home/mateus/IC/OnlineSfM/Dataset/" + (string)argv[1] + ".MOV";
        video_name = (string)argv[1]; 
        VideoCapture cap(path_to_vid);
        return sfm.runSfM(cap);
    }
}