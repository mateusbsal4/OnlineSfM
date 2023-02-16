#include <iostream>
#include <algorithm>
#include <string>
#include <numeric>
#define CERES_FOUND true
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <assert.h>
#include <ctype.h>
#include <tuple>
#include <cmath>
//#include <Eigen/Dense>
//#include <array>
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



using namespace std;
using namespace cv;
using namespace cv::utils::logging;
//using namespace Eigen;
using namespace xt;


//string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/primeira_sala1.MOV";
//string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/primeira_sala2.MOV";
//string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/primeira_sala3.mov";
//string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/segunda_sala1.mov";
//string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/segunda_sala2.MOV";
//string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/segunda_sala3.MOV";
//string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/segunda_sala4.MOV";
//string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/primeiro_quarto1.MOV";
//string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/primeiro_quarto2.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/primeiro_quarto3.MOV";    
string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/segundo_quarto.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_long/xadrez_cc.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_long/xadrez_cd.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_long/casa_cc.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/elef5.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_short/casa.MOV";




//THESE CRITERIA ARE SENSITIVE TO CHANGES
//TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
//Size subPixWinSize(10,10), winSize(31,31);            //example (standard OpenCV) criteria

TermCriteria termcrit(3,30,0.003);
Size subPixWinSize(10,10), winSize(15,15);           // tcc criteria

size_t initial_indexes_shape = 1;   
xarray<int> indexes({initial_indexes_shape});

//size_t initial_cloud_shape = 1;
//xarray<float> cloud({initial_cloud_shape});


//size_t initial_features_shape = 0;
//xarray<double> features({initial_features_shape});



VideoCapture cap(path_to_vid);            //post visualization
//VideoCapture cap(0);                        //online visualization

class StructureFromMotion{
    private:
        vector<vector<Point2f>> tracks;
        vector<vector<int>> masks;
        vector<Point2f> track, features, prev_points, points, new_points;
        Mat frame, color, image, gray, prevGray;
        vector<int> frame_numbers, mask;
        int frame_number;
        //vector<xarray<double>> Rs, Ts; 
        vector<Mat> Rs, Ts;       
        vector<Point3f> cloud = {};
        //Matx33f K = {4826.28455, 0.0, 1611.73703,
        //             0.0, 4827.31363, 1330.23261,
        //             0.0, 0.0, 1.0};

        Mat K = (Mat_<double>(3,3) << 4826.28455, 0, 1611.73703, 0, 4827.31363, 1330.23261, 0, 0, 1);
        const int skip_at_getter = 1; //skip_at_getter in tcc pipeline
        const int yield_period = 2;
        const LogTag TAG = LogTag("SfM", LOG_LEVEL_DEBUG);
        size_t frame_counter = 0;
        size_t i, j;
        size_t initial_counter = 0;
        const int MAX_COUNT = 300;
        //const int MAX_COUNT = 130;
        //const int MAX_COUNT = 100;      //tcc param
        const double closeness_threshold = 15;
        //const int min_features = 100;
        const int min_features = 100; 
        //const int min_features = 35; //tcc param
        bool needToInit = 1;
        //bool first_frame = 1;
        int start_index = 0;
        float error_threshold  = 50;
        int init_reconstruction_frames = 5;
        int error_calculation_frames = 5;
        bool is_init = 1;
        double ransac_probability = 0.999999;
        double essential_mat_threshold = 5;
        double distance_thresh = 500;

        bool use_epnp = 1;
        bool use_iterative_pnp = 0;
        int min_number_of_points = 5;

        bool use_five_pt_algorithm = 0;
        bool use_solve_pnp = 1;


        


    public:
        StructureFromMotion(){};


        void runSfM(){
            //vector<Point3f> cloud = {};
            //vector<Mat> Rs, Ts;
            for(;;){
                frame_counter ++;
                if(frame_counter%(skip_at_getter+1) !=0){continue;}
                tie(frame_number, track, mask) = feature_detector();
                tracks.push_back(track);                                    //this corresponds to init_reconstruction (before reconstruct) on tcc version
                masks.push_back(mask);
                frame_numbers.push_back(frame_number);
                if(tracks.size() < init_reconstruction_frames + error_calculation_frames){
                    continue;
                }
                if(is_init){
                    vector<Point3f> init_cloud = {};
                    vector<Mat> init_Rs, init_Ts;
                    initial_counter = 0;
                    std::vector<vector<Point2f>> init_tracks = std::vector<vector<Point2f>> (tracks.begin(), tracks.begin()+init_reconstruction_frames);
                    std::vector<vector<int>> init_masks = std::vector<vector<int>> (masks.begin(), masks.begin()+init_reconstruction_frames);               //remember that here some sort of update mechanism is necessary 
                    //std::vector<int> init_frame_numbers = std::vector<int> (frame_numbers.begin(), frame_numbers.begin()+init_reconstruction_frames);       //in order to allow the repetition of the init_reconstruction phase. For instance, 
                    assert(init_tracks.size() == init_masks.size());                                                                                        //sth like summing a "dropped_tracks = 1" integer to both ends of the iterator(this variable will depend on the reconstruction error)
                    cout << "III: " << initial_counter << endl;
                    tie(init_Rs, init_Ts, init_cloud) = init_reconstruction(init_tracks, init_masks, init_Rs, init_Ts, init_cloud);
                    is_init = 0;
                    Rs = init_Rs;
                    Ts = init_Ts;
                    cloud = init_cloud;
                    create_csv();
                    continue;
                }
                
                //cout << "Final init phase cloud: " << endl;
                //for (auto &point : cloud) {
                //    cout << point << endl;
                //}
                //cout << "Len of cloud: " << cloud.size() << endl;
                //for (auto &R : Rs) {
                //    cout << R << endl;
                //} 
                //for (auto &T : Ts) {
                //    cout << T << endl;
                //}
            }
        }

        std::tuple<int, vector<Point2f>, vector<int>> feature_detector(){
            cap >> frame;
            cout << "frame counter: " << frame_counter << endl;
            Mat image;
            CV_LOG_INFO(&TAG, "Detecting Features");
            color = frame;
            Mat mask = Mat::zeros(color.size(), color.type());
            Mat tracking_window;
            color.copyTo(image);
            cvtColor(image, gray, COLOR_BGR2GRAY);
            //cvtColor(color, gray, COLOR_BGR2GRAY);
            if( needToInit )
            {
                get_new_features(mask, image);
                needToInit = 0;
                start_index = amax(indexes)()+1;

                cout << "Start index: " << start_index << endl;
            }


            else if( !prevGray.empty() ){
                track_features(mask, image);
            }


            add(image, mask, tracking_window);
            //add(color, mask, tracking_window);
            imshow("LK Tracker", tracking_window);
            waitKey(10);

            std::swap(points, prev_points);
            cv::swap(prevGray, gray);
            if(prev_points.size() < min_features){
                needToInit = 1;
            }
            //cout << "INDEXES" << indexes << endl;
            if(frame_counter % yield_period ==0){
                vector<int> inds(indexes.begin(), indexes.end());
                return  std::make_tuple(frame_number, prev_points, inds);
            }

        }



        void get_new_features(Mat mask, Mat image){
            //goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);   //example (standard OpenCV) params - block size (3) is too small! This selects too many features
            //cornerSubPix(gray, new_points, subPixWinSize, Size(-1,-1), termcrit);
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
                indexes.resize({points.size()});
                indexes = arange(points.size());
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

            //cout << "INDEXES BEFORE TRACKING: " << indexes;
            auto xstatus = adapt(status, {status.size()});
            indexes = filter(indexes, xstatus);
            //cout << "INDEXES AFTER TRACKING: " << indexes;
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
            //cout << adapt(closeness_table.shape()) << endl;
            //cout << "CLOS TABLE " << closeness_table << endl;
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
            //cout << "NEW POINTS MASK " << new_points_mask << endl;
            //cout << "MASK SIZE " << new_points_mask.size() << endl;
            //cout << "MASK DIMENSION " << new_points_mask.dimension() << endl;
            //cout << "THERE ARE: " << ones_in_mask << " ONES IN MASK" << endl;
            auto old_features  = adapt(points, {n});
            auto new_features = adapt(new_points, {m});



            new_features = filter(new_features, new_points_mask);
            assert(ones_in_mask == new_features.size());
            auto new_indexes = arange(0, ones_in_mask)+start_index;

            auto features = xt::hstack(xtuple(old_features, new_features));
            //auto features = xt::hstack(xtuple(old_features, new_features));
            indexes = xt::hstack(xtuple(indexes, new_indexes));

            vector<Point2f> feats(features.begin(), features.end());
            points = feats;

            assert(points.size() ==indexes.size());
        }



        double norm(Point2f a, Point2f b){
            return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
        }



        tuple<vector<Mat>, vector<Mat>, vector<Point3f>> init_reconstruction(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud){
            //vector<Mat> Rs, Ts;
            Mat R, T;
            vector<Point3f> pts_3d;
            xarray<int> indexes;

            for(initial_counter = 0; initial_counter < init_reconstruction_frames; initial_counter++){      
                cout << "I N I T I A L " << endl;
                cout << "C O U N T E R " << initial_counter << endl;   
                vector<vector<Point2f>> init_tracks_sliced  = std::vector<vector<Point2f>> (init_tracks.begin(), init_tracks.begin()+initial_counter+1);                     
                vector<vector<int>> init_masks_sliced = std::vector<vector<int>> (init_masks.begin(), init_masks.begin()+initial_counter+1);                     
                cout << "SIZE OF INIT TRACKS: " << init_tracks_sliced.size() << endl;
                cout << cloud << endl;  
                if(cloud.empty()){
                    cout << "Entering" << endl;
                    if(init_masks_sliced.size() > 1 && init_tracks_sliced[0].size() >= 5){
                        tie(Rs, Ts, cloud) = five_pt_init(init_tracks_sliced, init_masks_sliced, Rs, Ts, cloud);
                        cout << "SIZE OF INIT TRACKS 2: " << init_tracks_sliced.size() << endl;
                        //cout << Rs[0] << endl;
                        assert(Rs.size() == 2);
                    }
                    //cout << "RSSSSS: " << Rs[0] << endl;
                    continue;
                }
                //assert(Rs.size() == 2);
                cout << "Size of Rs" << Rs.size() << endl;
                cout << "SIZE OF INIT TRACKS 3: " << init_tracks_sliced.size() << endl;
                tie(R, T, pts_3d, indexes) = calculate_projection(Rs[Rs.size()-1], Ts[Ts.size()-1], init_tracks_sliced, init_masks_sliced, cloud);
                if(!pts_3d.empty()){
                    cout << "Entering? " << endl;
                    add_points_to_cloud(pts_3d, indexes, cloud);
                }
                Rs.push_back(R);
                Ts.push_back(T);
            }
            return make_tuple(Rs, Ts, cloud);  
        }
//




        tuple<Mat, Mat, vector<Point3f>, xarray<int>> calculate_projection(Mat prev_R, Mat prev_T, vector<vector<Point2f>> tracks, vector<vector<int>> masks, vector<Point3f> cloud){
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
            cout << "R 1 " << R << endl;
            cout << "T 1 " << T << endl;
            cout << "prev R: " << prev_R << endl;
            cout << "prev T: " << prev_T << endl;
            //triangulate(prev_R, prev_T, R, T, )
            vector<Point3f> points_3d = triangulate(prev_R.t(), -(prev_R.t())*prev_T, R.t(), -(R.t())*T, track_pair);           //careful here!! Cv operations wont work with empty matrices
            return make_tuple(R, T, points_3d, pair_mask);  
        }

        
        tuple<Mat, Mat> solve_pnp(vector<Point2f> track, vector<int> mask, Mat R_est, Mat T_est, vector<Point3f> cloud){
            Mat R, T;
            if(use_epnp){
                tie(R, T) = solve_pnp_(track, mask, SOLVEPNP_EPNP, R_est, T_est, cloud);
                cout << "R 2 " << R << endl;
                R_est = R;
                T_est = T;
            }   
            if(use_iterative_pnp){
                //R = xarray_to_mat_elementwise(xR);
                //T = xarray_to_mat_elementwise(xT);
                cout << "R 3 " << R << endl;
                tie(R, T) = solve_pnp_(track, mask, SOLVEPNP_ITERATIVE, R_est, T_est, cloud);
            }
            cout << "R 4: " << R << endl;
            return make_tuple(R, T);
        }

        tuple<Mat, Mat> solve_pnp_(vector<Point2f> track_slice, vector<int> track_mask, cv::SolvePnPMethod method, Mat R, Mat T, vector<Point3f> cloud){
            cout << "Getting here? " << endl;
           bool use_extrinsic_guess = (!R.empty() && !T.empty()) ? 1 : 0;
           xarray<int> xcloud_mask = get_not_nan_index_mask(cloud); 
           //cout << "XCLOUD MASK: " << xcloud_mask << endl;
           vector<int> cloud_mask(xcloud_mask.begin(), xcloud_mask.end());
           xarray<int> intersection_mask = get_intersection_mask(cloud_mask, track_mask);
           //cout << "Intersection mask: " << intersection_mask << endl; 
           cout << "Size of cloud: " << cloud.size() << endl;

           xarray<int> xtrack_mask = adapt(track_mask, {track_mask.size()});
           xarray<bool> track_bool_mask = isin(xtrack_mask, intersection_mask);
           cout << "Size of intersection mask: " << intersection_mask.size() << endl;
           if(intersection_mask.size()<min_number_of_points){
                return make_tuple(R,T);
           }
           cout << "Here? " << endl;
           tie(R, T) = invert_reference_frame(R,T);
           cout << "Intersection mask: " << intersection_mask << endl;
           vector<Point3f> pts_cloud;
           for(i = 0; i<cloud.size(); i++){
            if(std::find(intersection_mask.begin(), intersection_mask.end(), i) != intersection_mask.end()){
                pts_cloud.push_back(cloud[i]);
            }
           }
           cout << "Size of cloud after filtering: " << pts_cloud.size() << endl;
           cloud = pts_cloud;
           auto xtrack_slice = adapt(track_slice, {track_slice.size()});
           xtrack_slice = filter(xtrack_slice, track_bool_mask);
           vector<Point2f> img_points(xtrack_slice.begin(), xtrack_slice.end());
           Mat rvec, distCoeffs;
           if(!R.empty()){
                Rodrigues(R, rvec);
           }
           cout << "rvec: " << rvec << endl;
           cout << "Image points 0: " << img_points[0] << endl;
           assert(cloud.size() == img_points.size());
           solvePnP(cloud, img_points, K, distCoeffs, rvec, T, use_extrinsic_guess, method);
           cout << "rvec after call: " << rvec << endl;
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
                return make_tuple(null_Rs, null_Rs, cloud);
            }
            vector<Point3f> points3d;
            //if(R.dimension()==2 && T.dimension()==2 && Rs[0].dimension()==2 && Ts[0].dimension()==2){
            //    points3d = triangulate(transpose(Rs[0]), -linalg::dot(transpose(Rs[0]), Ts[0]), transpose(R), -linalg::dot(transpose(R),T), track_pair);                                              //reconverting motion matrices to relative frame coordinate system is required by the OpenCV function
            //}
            cout << "PAIR MASK: " << pair_mask << endl;
            cout << "LEN PAIR MASK: "<< pair_mask.size();
            cout << "MAX PAIR MASK: " << amax(pair_mask)() +1 << endl;
            assert(track_pair[0].size() == pair_mask.size());
            assert(pair_mask.size() == amax(pair_mask)() + 1);
            //assert(track_pair[0].size() == pair_mask.size() == amax(pair_mask)() + 1);
            points3d = triangulate(Rs[0].t(), -(Rs[0].t())*Ts[0], R.t(), -(R.t())*T, track_pair);                                              //reconverting motion matrices to relative frame coordinate system is required by the OpenCV function
            assert(points3d.size() == pair_mask.size());
            cloud = points_to_cloud(points3d, pair_mask); 
            Rs.push_back(R);
            Ts.push_back(T);
            return make_tuple(Rs, Ts, cloud);

        }

        tuple<Mat, Mat> five_pt(vector<vector<Point2f>> track_pair, xarray<int> pair_mask, Mat prev_R, Mat prev_T){
            Mat E, five_pt_mask, mask, R, T; 
            //tie(track_pair, pair_mask)  = get_last_track_pair(init_tracks, init_masks);
            E = findEssentialMat(track_pair[0], track_pair[1], K,  RANSAC, ransac_probability, essential_mat_threshold, five_pt_mask);         //error was here
            five_pt_mask.copyTo(mask);
            recoverPose(E, track_pair[0], track_pair[1], K, R, T, distance_thresh, five_pt_mask);

            cout << "Mat R = " << endl << " " << R << endl << endl;
            cout << "Mat T = " << endl << " " << T << endl << endl;
            tie(R, T) = invert_reference_frame(R,T);
            //cout << "Dimension OF FILLED R: " << R.dimension() << endl;
            //cout << "Dimension OF FILLED T: " << T.dimension() << endl;
            printf("Filled R dimensions: %s %dx%d \n", type2str( R.type() ).c_str(), R.rows, R.cols );
            printf("Filled T dimensions: %s %dx%d \n", type2str( T.type() ).c_str(), T.rows, T.cols );
            printf("Filled R0 dimensions: %s %dx%d \n", type2str( prev_R.type() ).c_str(), prev_R.rows, prev_R.cols );
            printf("Filled T0 dimensions: %s %dx%d \n", type2str( prev_T.type() ).c_str(), prev_T.rows, prev_T.cols );
            //cout << "Dimension of first R: " << prev_R.dimension() << endl;
            //cout << "Dimension of first T: " << prev_T.dimension() << endl;

            cout << "Transposed R: " << R << endl;
            cout << "Linalg T: " << T << endl;
            //assert(Rs.size()-1 ==0);
            //assert(Ts.size()-1 ==0);
            tie(R, T) = compose_rts(R, T, prev_R, prev_T);
            return make_tuple(R,T);

        }


        vector<Point3f> points_to_cloud(vector<Point3f> points3d, xarray<int> pair_mask){
            cout << "Max track_mask: " << amax(pair_mask)() +1 << endl;
            auto points_3d = adapt(points3d, {points3d.size()});
            cout << "XPointstocloud Dimension: " << points_3d.dimension() << endl;
            cout << "XPointstocloud Size: " << points_3d.size() << endl;
            cout << "XPointstocloud SHape: " <<  adapt(points_3d.shape()) << endl;
            //auto xcloud = full_like(points_3d, {0.0,0.0,0.0});
            //size_t szxcloud = amax(pair_mask)() +1;
            xarray<Point3f> xcloud = {{0.0,0.0,0.0}};
            xcloud.resize({amax(pair_mask)() +1});
            xcloud = all_nan(xcloud);
            cout << "XCLOUD " << endl;
            cout << "RESIZED " << xcloud << endl;
            //for(i=0; i<xcloud.size())
            cout << "Xcloud Dimension: " << xcloud.dimension() << endl;
            cout << "Xcloud Size: " << xcloud.size() << endl;
            cout << "Xcloud SHape: " <<  adapt(xcloud.shape()) << endl;
            cout << "Xcloud 2nd el: " << xcloud(1) << endl;
            //assert(xcloud.size() == points3d.size());
            //xcloud = filter(xcloud, pair_mask);
            xcloud = points_3d;
            assert(xcloud.size() == points_3d.size());
            assert(adapt(xcloud.shape()) == adapt(points_3d.shape()));            
            assert(xcloud.dimension() == points_3d.dimension());
            //xcloud.filter(pair_mask) = points_3d;
            cout << "Xcloud:" << xcloud;
            cout << "Indexes:" << pair_mask;
            vector<Point3f> points_cloud(xcloud.begin(), xcloud.end());
            //cloud = points_cloud;
            return points_cloud;
        }


        void add_points_to_cloud(vector<Point3f> points_3d, xarray<int> indexes, vector<Point3f> cloud){
            assert(!cloud.empty());
            xarray<int> cloud_mask = get_not_nan_index_mask(cloud);
            xarray<int> new_points_mask = setdiff1d(indexes, cloud_mask);
            xarray<Point3f> xcloud = adapt(cloud, {cloud.size()});
            if(amax(indexes)() > cloud.size()){
                xarray<Point3f> new_cloud = {{0.0,0.0,0.0}};
                new_cloud.resize({2*amax(indexes)()});
               //filter(new_cloud, cloud_mask) = filter(xcloud, cloud_mask);
               new_cloud = double_filter(new_cloud, xcloud, cloud_mask);
               xcloud.resize({2*amax(indexes)()});
               xcloud = new_cloud;
            }
            cout << "XCLOUD BEFORE: " << xcloud << endl;
            cout << "Len xcloud before: " << xcloud.size() << endl;
            if(new_points_mask.size() >= 1){
                auto xpoints_3d = adapt(points_3d, {points_3d.size()});
                xarray<Point3f> new_points = filter(xpoints_3d, isin(indexes, new_points_mask));
                cout << "NEW POINTS MASK: " << new_points_mask;
                for(i = 0; i< new_points_mask.size(); i++){
                    int index = new_points_mask(i);
                    xcloud(index) = new_points(i);
                }
            }    
            cout << "XCLOUD AFTER: " << xcloud << endl;
            cout << "Len xcloud after: " << xcloud.size() << endl;
            vector<Point3f> points_cloud(xcloud.begin(), xcloud.end());
            cloud = points_cloud;
            
        }

        xarray<Point3f> double_filter(xarray<Point3f> first_vec, xarray<Point3f> second_vec, xarray<int> mask1, xarray<int> mask2 = {}){
            cout << "LEN FIRST VEC BEFORE: "<< first_vec.size() << endl;
            for(int index: mask1){
                first_vec(index) = second_vec(index);
            }
            cout << "LEN FIRST VEC AFTER: " << first_vec.size() << endl;
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
            cout << "K in xarray = " << xK << endl;
            P1 = linalg::dot(xK, hstack(xtuple(xR1, xT1)));
            P2 = linalg::dot(xK, hstack(xtuple(xR2, xT2)));
            cout << "R1 = " << xR1 << endl;
            cout << "T1 = " << xT1 << endl;
            cout << "[R1|T1] = " << hstack(xtuple(xR1, xT1));
            cout << "R2 = " << xR2 << endl;
            cout << "T2 = " << xT2 << endl;
            cout << "[R2|T2] = " << hstack(xtuple(xR2, xT2));
            cout << "Proj matrix 1: " << P1 << endl;
            cout << "Proj matrix 2: " << P2 << endl;
            Mat P_1, P_2, points4d;
            P_1 = xarray_to_mat_elementwise(P1);
            P_2 = xarray_to_mat_elementwise(P2);
            cout << "X Proj matrix1: " << P_1 << endl;
            cout << "X Proj matrix2: " << P_2 << endl;
            assert(track_pair[0].size() == track_pair[1].size());
            //assert(track_pair[0].size()) == pair_mask.size();
            triangulatePoints(P_1, P_2, track_pair[0], track_pair[1], points4d);
            printf("Matrix: %s %dx%d \n", type2str( points4d.type() ).c_str(), points4d.rows, points4d.cols );
            assert(points4d.cols == track_pair[0].size());
            assert(points4d.rows != 0 && points4d.cols !=0);
            //cout << "cols: " << points4d.cols()<< endl;
            //cout << "rows: " << points4d.rows() << endl;
            cout << "MAT PTS4D FIRST ELEMENT: " << points4d.at<float>(0,0) << endl;
            cout << "MAT PTS4D FIRST ELEMENT: " << points4d.at<float>(1,0) << endl;
            cout << "MAT PTS4D FIRST ELEMENT: " << points4d.at<float>(2,0) << endl;
            cout << "MAT PTS4D FIRST ELEMENT: " << points4d.at<float>(3,0) << endl;
            //cout << "MAT PTS4D FIRST ELEMENT: " << points4d.at<float>(4,0) << endl;
            points4d = points4d.t();
            printf("Matrix: %s %dx%d \n", type2str( points4d.type() ).c_str(), points4d.rows, points4d.cols );
            cout << "MAT TRANSPOSED FIRST ELEMENT: " << points4d.at<float>(0,0) << endl;
            cout << "MAT TRANSPOSED FIRST ELEMENT: " << points4d.at<float>(0,1) << endl;
            cout << "MAT TRANSPOSED FIRST ELEMENT: " << points4d.at<float>(0,2) << endl;
            cout << "MAT TRANSPOSED FIRST ELEMENT: " << points4d.at<float>(0,3) << endl;    
            Mat pts4d = points4d.reshape(4,1);
            printf("Points homogeneous: %s %dx%d \n", type2str( pts4d.type() ).c_str(), pts4d.rows, pts4d.cols );
            cout << "Size of features: " << track_pair[0].size() << endl;
            //Vec2f& elem = pts4d.at<Vec2f>( 0 , 0 );
            cout << "reshaped 0: " << pts4d.at<Vec4f>(0,0)[0] << endl;
            cout << "reshaped 1: " << pts4d.at<Vec4f>(0,0)[1] << endl;
            cout << "reshaped 2: " << pts4d.at<Vec4f>(0,0)[2] << endl;
            cout << "reshaped 3: " << pts4d.at<Vec4f>(0,0)[3] << endl;

            //Mat points3d;
            convertPointsFromHomogeneous(pts4d, points3d);
            cout << "SIZE OF POINTS 3D: " << points3d.size() << endl;
            cout << "First 3d point: " << points3d[0] << endl;
            cout << "(x)" << points3d[0].x << " = " <<  (pts4d.at<Vec4f>(0,0)[0])/(pts4d.at<Vec4f>(0,0)[3]) << "(X/W)?" << endl;
            cout << "(y)" << points3d[0].y << " = " <<  (pts4d.at<Vec4f>(0,0)[1])/(pts4d.at<Vec4f>(0,0)[3]) << "(Y/W)?" << endl;
            cout << "(z)" << points3d[0].z << " = " <<  (pts4d.at<Vec4f>(0,0)[2])/(pts4d.at<Vec4f>(0,0)[3]) << "(Z/W)?" << endl;
            return points3d;
        }


        string type2str(int type) {
          string r;

          uchar depth = type & CV_MAT_DEPTH_MASK;
          uchar chans = 1 + (type >> CV_CN_SHIFT);

          switch ( depth ) {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
          }

          r += "C";
          r += (chans+'0');

          return r;
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

        //cv::Mat xarray_to_mat(xt::xarray<double> xarr)
        //{
        //    cv::Mat mat (xarr.shape()[0], xarr.shape()[1], CV_64FC1, xarr.data(), 0);             //this is not working
        //    return mat;
        //}
        cv::Mat xarray_to_mat_elementwise(xt::xarray<double> xarr)
        {
            int ndims = xarr.dimension();
            assert(ndims == 2  && "can only convert 2d xarrays");
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

        xt::xarray<float> mat_to_xarray_float(cv::Mat mat)
        {
            xt::xarray<float> res = xt::adapt(
                (float*) mat.data, mat.cols * mat.rows, xt::no_ownership(), std::vector<std::size_t> {mat.rows, mat.cols});
            return res;
        }


        xt::xarray<double> mat_to_xarray(cv::Mat mat)
        {
            xt::xarray<double> res = xt::adapt(
                (double*) mat.data, mat.cols * mat.rows, xt::no_ownership(), std::vector<std::size_t> {mat.rows, mat.cols});
            return res;
        }

        //xt::xarray<int> mat_mask_to_xarray(cv::Mat mat)
        //{
        //    xt::xarray<int> res = xt::adapt(
        //        (int*) mat.data, mat.cols * mat.rows, xt::no_ownership(), std::vector<std::size_t> {mat.rows, mat.cols});
        //    return res;
        //}

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
            cout << "Size of init tracks: " << init_tracks.size() << endl;
            vector<Point2f> init_track1 = init_tracks[init_tracks.size()-1];
            vector<Point2f> init_track2 = init_tracks[init_tracks.size()-2];
            vector<int> init_mask1 = init_masks[init_masks.size()-1];
            vector<int> init_mask2 = init_masks[init_masks.size()-2];

            //auto xpair_mask = get_intersection_mask(init_mask1, init_mask2);

            auto xpair_mask = get_intersection_mask(init_mask2, init_mask1);
            assert(get_intersection_mask(init_mask1, init_mask2) == get_intersection_mask(init_mask2, init_mask1));
            auto xmask1 = adapt(init_mask1, {init_mask1.size()});
            cout << "mask 1: " << xmask1 << endl;
            auto xmask2 = adapt(init_mask2, {init_mask2.size()});
            cout << "mask 2: " << xmask2 << endl;
            cout << "pair mask: " << xpair_mask << endl;


            auto xtrack1 = adapt(init_track1, {init_track1.size()});
            //cout << "TRACK 1: " << xtrack1 << endl;
            auto xtrack2 = adapt(init_track2, {init_track2.size()});
            //cout << "TRACK 2: " << xtrack2 << endl;

            //assert(xtrack1.size()==isin(xmask1, xpair_mask).size());
            cout << "isin_mask1: " << isin(xmask1, xpair_mask);
            cout << "isin_mask2: " << isin(xmask2, xpair_mask);
            auto xtrack_pair0 = filter(xtrack1, isin(xmask1, xpair_mask));
            auto xtrack_pair1 = filter(xtrack2, isin(xmask2, xpair_mask));

            vector<Point2f> track_pair0(xtrack_pair0.begin(), xtrack_pair0.end());
            vector<Point2f> track_pair1(xtrack_pair1.begin(), xtrack_pair1.end());


            //vector<vector<Point2f>> track_pair{track_pair0, track_pair1};
            vector<vector<Point2f>> track_pair{track_pair1, track_pair0};
            //vector<vector<Point2f>> track_pair;
            //track_pair.push_back(track_pair0);
            //track_pair.push_back(track_pair1);


            return make_tuple(track_pair, xpair_mask);

        }


        void create_csv(){
            ofstream fout;
            fout.open("init_cloud.txt");
            fout << "cloud = {";
            for (i = 0; i<cloud.size()-1; i++) {
                fout << "{" << cloud[i].x << ", " << cloud[i].y <<  ", " << cloud[i].z << "}" << ", ";
            }
            fout << "{" << cloud[cloud.size()-1].x << ", " << cloud[cloud.size()-1].y <<  ", " << cloud[cloud.size()-1].z << "}}" << endl;
            for (i = 0; i< 5; i++){
                fout << "{R" << i << ", " << "T" << i << "} = " << "{{{" << Rs[i].at<double>(0, 0) << ", " << Rs[i].at<double>(0, 1) << ", " << Rs[i].at<double>(0, 2) << "}, ";
                fout << "{" << Rs[i].at<double>(1, 0) << ", " << Rs[i].at<double>(1, 1) << ", " << Rs[i].at<double>(1, 2) << "}, ";
                fout << "{" << Rs[i].at<double>(2, 0) << ", " << Rs[i].at<double>(2, 1) << ", " << Rs[i].at<double>(2, 2) << "}}";
                fout << ", {" << Ts[i].at<double>(0, 0) << ", " << Ts[i].at<double>(1, 0) << ", " << Ts[i].at<double>(2, 0) << "}} " << endl;
            }
            //fout << "Rs = {{" << 
            fout.close();

        }


};

int main(){
    StructureFromMotion sfm;
    sfm.runSfM();
    return 0;
}