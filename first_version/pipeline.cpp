#include <iostream>
#include <algorithm>
#include <string>
#include <numeric>
#define CERES_FOUND true
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
        //float error_threshold = 1.5;      //note: this is not an absolute value, sometimes it is impossible to have a so low threshold for a given example
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
        


        


    public:
        StructureFromMotion(){}


        int runSfM(VideoCapture cap){
            viz::Viz3d window("SfM Visualization");
            window.setWindowSize(Size(2000,2000));
            //window.setWindowPosition(Point(150,150));
            //viz::Viz3d myWindow("Coordinate Frame");
            //window.setWindowSize(Size(1000,1000));
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
                    //MEMORY ERROR FOR XADREZ_CC occurs somewhere from this line
                    if(!cloud_3d.empty()){
                        error = calculate_init_error(error_calc_tracks, error_calc_masks, cloud_3d);        //occurs here
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
                
                viz::WCloud cloud_widget(cloud_3d, viz::Color::white());
                //Affine3d viewer_pose(Vec3d(0, 0, -1000));
                //float scale_factor = 2.0; 
                //cv::Affine3d scale_transform = cv::viz::makeTransformToGlobal(Vec3f(0.0f,-2.0f,0.0f), Vec3f(-2.0f,0.0f,0.0f), Vec3f(0.0f,0.0f,-2.0f));
                //cloud_widget.setPose(scale_transform * cloud_widget.getPose());
                cloud_widget.setRenderingProperty(viz::POINT_SIZE, 3.0);
                window.showWidget("point_cloud", cloud_widget);
                if(global_Rs.size() > 0){
                    for(int path_counter = path.size(); path_counter<global_Rs.size(); path_counter ++){
                        path.push_back(Affine3d(global_Rs[path_counter], global_Ts[path_counter]));
                    }
                    window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::PATH, 0.1, viz::Color::blue()));
                    window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K_viz, 0.1, viz::Color::red()));
                    //window.setViewerPose(viewer_pose);                    
                }
                //break;  //testing purpose: init error read
                vector<vector<Point2f>> remaining_tracks(tracks.end() - error_calculation_frames-1, tracks.end());
                vector<vector<int>> remaining_masks(masks.end() - error_calculation_frames-1, masks.end());                       
                vector<int> remaining_frame_numbers(frame_numbers.end()-error_calculation_frames-1, frame_numbers.end()); 
                
                tie(global_Rs, global_Ts, cloud_3d) = reconstruct(remaining_tracks, remaining_masks, global_Rs, global_Ts, cloud_3d); 
                window.spinOnce(1, true);          
            }
            cout << "Ended " << endl;
            while(1){window.spinOnce(1, true);}     //keeps the window opened after the pipeline has ended
            //window.spin();
            return 0;
        }



        tuple<vector<Mat>, vector<Mat>, vector<Point3f>> reconstruct(vector<vector<Point2f>> tracks, vector<vector<int>> masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud){
            Mat R, T; 
            vector<Point3f> new_pts;
            xarray<int> new_pt_indexes;
            tie(R, T, new_pts, new_pt_indexes) = calculate_projection(Rs[Rs.size() - 1], Ts[Ts.size() - 1], tracks, masks, cloud);
            //cout << "NEW PT INDEXES: " << new_pt_indexes << endl;
            if(!new_pts.empty()){
                cloud = add_points_to_cloud(new_pts, new_pt_indexes, cloud);
                //add_points_to_cloud(new_pts, new_pt_indexes, cloud);
            }
            Rs.push_back(R);
            Ts.push_back(T);
            return make_tuple(Rs, Ts, cloud);
        }

        bool check_end(VideoCapture cap){
            cap >> frame;
            return (frame.empty())?1:0;
        }


        std::tuple<int, vector<Point2f>, vector<int>> feature_detector(){
            cout << "frame counter: " << frame_counter << endl;
            Mat image;
            //CV_LOG_INFO(&TAG, "Detecting Features");
            color = frame;
            Mat mask = Mat::zeros(color.size(), color.type());
            Mat tracking_window;
            color.copyTo(image);
            cvtColor(image, gray, COLOR_BGR2GRAY);
            if( needToInit )
            {
                get_new_features(mask, image);
                needToInit = 0;
                //cout << "INDEXES_2D " << indexes_2d << endl;
                start_index = amax(indexes_2d)()+1;
                cout << amax(indexes_2d)();
                //cout << "Start index: " << start_index << endl;
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
            //cout << "INDEXES" << indexes_2d << endl;
            if(frame_counter % yield_period ==0){
                //cout << "Last of indexes 2d: " << *indexes_2d.end() << endl;
                //cout << "Last of indexes 2d: " << *(indexes_2d.begin() + indexes_2d.size()) << endl;      //testing purposes 
                //if(frame_counter == 20){while(1);}
                vector<int> inds(indexes_2d.begin(), indexes_2d.end());         //this works
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

            //cout << "INDEXES BEFORE TRACKING: " << indexes_2d;
            auto xstatus = adapt(status, {status.size()});
            indexes_2d = filter(indexes_2d, xstatus);
            //cout << "indexes_2d AFTER TRACKING: " << indexes_2d;
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
            //cout << "Prior indexes: " << indexes_2d;
            //cout << "New indexes: " << new_indexes;
            auto features = xt::hstack(xtuple(old_features, new_features));
            //auto features = xt::hstack(xtuple(old_features, new_features));
            indexes_2d = xt::hstack(xtuple(indexes_2d, new_indexes));

            vector<Point2f> feats(features.begin(), features.end());            //this IS working

            //cout << "Last theoretical point " << *features.end() << endl;
            //cout << "Last real point: " << *(features.begin() + features.size()-1) << endl;      //testing purposes 
            //cout << "Xcloud mask: " << features << endl;
            //for(int r = 0; r< feats.size(); r++){
            //    cout << "CLoud mask " << r << ": " << feats[r] << endl;
            //}
            //if(frame_counter==40){while(1);}     //testing purposes

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
            //assert(error_calc_masks.size() == error_calc_frame_numbers.size());
            //assert(error_calc_frame_numbers.size() == 5);
            for(c =0; c<error_calc_masks.size(); c++){
                tie(R, T) = solve_pnp(error_calc_tracks[c], error_calc_masks[c], empty_R, empty_T, pt_cloud);
                error_calc_Rs.push_back(R);
                error_calc_Ts.push_back(T);             
            }
            //return 1.0f;
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
            vector<int> cloud_mask(xcloud_mask.begin(), xcloud_mask.end());     //this IS working 

            //cout << "Last theoretical point " << *xcloud_mask.end() << endl;
            //cout << "Last real point: " << *(xcloud_mask.begin() + xcloud_mask.size()-1) << endl;      //testing purposes 
            //cout << "Xcloud mask: " << xcloud_mask << endl;
            //for(int m = 0; m< cloud_mask.size(); m++){
            //    cout << "CLoud mask " << m << ": " << cloud_mask[m] << endl;
            //}
            //if(frame_counter==40){while(1);}     //testing purposes


            vector<float> errors;
            for(c= 0 ; c< ec_tracks.size(); c++){
                vector<Point3f> filtered_cloud;
                R = ec_Rs[c];
                T = ec_Ts[c];
                original_track = ec_tracks[c];
                track_mask = ec_masks[c];
                xtrack_mask = adapt(track_mask, {track_mask.size()});    


                xarray<int> intersection_mask = get_intersection_mask(cloud_mask, track_mask);
                cout << "size of intersct mask" << intersection_mask.size() << endl;
                cout << "xtrack_mask " << xtrack_mask << endl;      
                auto track_bool_mask = isin(xtrack_mask, intersection_mask);
                //cout << "track bool mask: " << track_bool_mask << endl;
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
                //cout << "Size of cloud mask: " << cloud_mask.size() << endl;
                //cout << "Track bool mask: " << track_bool_mask.size() << endl;
                //cout << filtered_cloud.size() << endl;
                //cout << "Projected pts (originated from cloud): " << xprojected_pts.size() << endl;
                assert(intersection_mask.size() == xprojected_pts.size());
                //cout << "Original pts: " << filter(xoriginal_pts, track_bool_mask) << endl;
                //cout << "Projected pts: " << xprojected_pts << endl;
                assert(filter(xoriginal_pts, track_bool_mask).size() == xprojected_pts.size());
                auto xdelta = filter(xoriginal_pts, track_bool_mask) - xprojected_pts;
                //cout << "Delta: " << xdelta << endl; 
                //cout << "Size of delta: " << xdelta.size() << endl;
                assert(xdelta(0).x == filter(xoriginal_pts, track_bool_mask)(0).x - xprojected_pts(0).x);
                assert(xdelta(0).y == filter(xoriginal_pts, track_bool_mask)(0).y - xprojected_pts(0).y);
                //vector<Point2f> delta(xdelta.begin(), xdelta.end());
                vector<float> errors_per_frame;
//
                for(i = 0; i< xdelta.size(); i++){
                    errors_per_frame.push_back(sqrt((xdelta(i).x)*(xdelta(i).x)+(xdelta(i).y)*(xdelta(i).y)));
                    //cout << "errors_per_frame[" << i << "] = " << errors_per_frame[i] << endl;
                }
                errors.push_back(my_mean(errors_per_frame));    
            }
            //cout << "Final error: " << my_mean(errors) << endl;
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
                    //cout << "Entering" << endl;
                    if(init_masks_sliced.size() > 1 && init_tracks_sliced[0].size() >= 5){
                        tie(Rs, Ts, cloud) = five_pt_init(init_tracks_sliced, init_masks_sliced, Rs, Ts, cloud);
                        assert(Rs.size() == 2);
                    }   
                    continue;
                }

                //assert(Rs.size() == 2);
                tie(R, T, pts_3d, indexes) = calculate_projection(Rs[Rs.size()-1], Ts[Ts.size()-1], init_tracks_sliced, init_masks_sliced, cloud);
                if(!pts_3d.empty()){
                    cloud = add_points_to_cloud(pts_3d, indexes, cloud); 
                    //add_points_to_cloud(pts_3d, indexes, cloud); 
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
            //cout << "T 1 " << T << endl;
            //cout << "prev R: " << prev_R << endl;
            //cout << "prev T: " << prev_T << endl;
            //assert(prev_R.empty() && !prev_T.empty() && !R.empty() && !T.empty());
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
                //R = xarray_to_mat_elementwise(xR);
                //T = xarray_to_mat_elementwise(xT);
                tie(R, T) = solve_pnp_(track, mask, SOLVEPNP_ITERATIVE, R_est, T_est, cloud);
            }
            cout << "R 4: " << R << endl;
            cout << "T : " << T << endl;
            return make_tuple(R, T);
        }

        tuple<Mat, Mat> solve_pnp_(vector<Point2f> track_slice, vector<int> track_mask, cv::SolvePnPMethod method, Mat R, Mat T, vector<Point3f> cloud){
           bool use_extrinsic_guess = (!R.empty() && !T.empty()) ? 1 : 0;
           xarray<int> xcloud_mask = get_not_nan_index_mask(cloud); 
           vector<int> cloud_mask(xcloud_mask.begin(), xcloud_mask.end());
           xarray<int> intersection_mask = get_intersection_mask(cloud_mask, track_mask);
           //cout << "Intersection mask: " << intersection_mask << endl; 
           xarray<int> xtrack_mask = adapt(track_mask, {track_mask.size()});
           xarray<bool> track_bool_mask = isin(xtrack_mask, intersection_mask);
           if(intersection_mask.size()<min_number_of_points){
                return make_tuple(R,T);
           }
           tie(R, T) = invert_reference_frame(R,T);
           //cout << "Intersection mask: " << intersection_mask << endl;
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
           //cout << "rvec: " << rvec << endl;
           //cout << "Image points 0: " << img_points[0] << endl;
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


            //cout << "Last theoretical point " << *xcloud.end() << endl;
            //cout << "Last real point: " << *(xcloud.begin() + xcloud.size()-1) << endl;      //testing purposes 
            //cout << "Xcloud mask: " << xcloud << endl;
            //for(int t = 0; t< points_cloud.size(); t++){
            //    cout << "Points cloud " << t << ": " << points_cloud[t] << endl;
            //}
            //if(frame_counter==40){while(1);}     //testing purposes

            return points_cloud;
        }

        //add_points_to_cloud(vector<Point3f> points_3d, xarray<int> indexes, vector<Point3f> cloud){
        vector<Point3f> add_points_to_cloud(vector<Point3f> points_3d, xarray<int> indexes, vector<Point3f> cloud){   //NOTE: this function only adds new points to the cloud, it does not replace any point that has already been triangulated.
            assert(points_3d.size() == indexes.size());                                                       
            assert(!cloud.empty());                                                                        
            xarray<int> cloud_mask = get_not_nan_index_mask(cloud);                                 
            xarray<int> new_points_mask = setdiff1d(indexes, cloud_mask);
            //cout << "t3" << endl;
            cout << "New pts mask: " << new_points_mask << endl;
            cout << "CLoud mask: " << cloud_mask << endl;
            cout << "Indexes: " << indexes << endl;            
            xarray<Point3f> xcloud = adapt(cloud, {cloud.size()}); 
            cout << "Cloud: " << xcloud << endl;   
            cout << "Pts 3d: " << endl;
            for(int i = 0; i < points_3d.size(); i++){
                cout << points_3d[i] << endl;
            }
            //cout << "XCLOUD BEFORE: " << xcloud << endl;             
            if(amax(indexes)() > cloud.size() -1){
                xarray<Point3f> new_cloud = {{0.0,0.0,0.0}};   
                new_cloud.resize({2*amax(indexes)()});
                new_cloud = all_nan(new_cloud);
                new_cloud = double_filter(new_cloud, xcloud, cloud_mask);
                xcloud.resize({2*amax(indexes)()});
                xcloud = new_cloud;
               //cout << "AUGMENTED CLOUD:" << xcloud << endl;
            }   
            if(new_points_mask.size() >= 1){
                auto xpoints_3d = adapt(points_3d, {points_3d.size()});
                xarray<Point3f> new_points = filter(xpoints_3d, isin(indexes, new_points_mask));
                for(int i = 0; i< new_points.size(); i++){                //error is inside this if statement
                    int index = new_points_mask(i);
                    cout << "Index " << index << endl;  
                    xcloud(index) = new_points(i);
                    //cout << ""
                }
            }  
            //cout << "Len xcloud after: " << xcloud.size() << endl;
            vector<Point3f> points_cloud(xcloud.begin(), xcloud.end());  

            cout << "Last theoretical point " << *xcloud.end() << endl;
            cout << "Last real point: " << *(xcloud.begin() + xcloud.size()-1) << endl;      //testing purposes 
            //cout << "Xcloud: " << xcloud << endl; 
            //for(int u = 0; u< points_cloud.size(); u++){
            //    cout << "Points cloud " << u << ": " << points_cloud[u] << endl;
            //}
            //if(frame_counter==40){while(1);}     //testing purposes
            cout << "Points cloud: " << points_cloud << endl;
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
            //assert(track_pair[0].size()) == pair_mask.size();
            triangulatePoints(P_1, P_2, track_pair[0], track_pair[1], points4d);
            //printf("Matrix: %s %dx%d \n", type2str( points4d.type() ).c_str(), points4d.rows, points4d.cols );
            assert(points4d.cols == track_pair[0].size());
            assert(points4d.rows != 0 && points4d.cols !=0);
            points4d = points4d.t();
            //printf("Matrix: %s %dx%d \n", type2str( points4d.type() ).c_str(), points4d.rows, points4d.cols );
            Mat pts4d = points4d.reshape(4,1);
            //printf("Points homogeneous: %s %dx%d \n", type2str( pts4d.type() ).c_str(), pts4d.rows, pts4d.cols );
            //Mat points3d;
            convertPointsFromHomogeneous(pts4d, points3d);
            //cout << "SIZE OF POINTS 3D: " << points3d.size() << endl;
            //cout << "First 3d point: " << points3d[0] << endl;
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


        float my_mean(std::vector<float> const& v)
        {
            if(v.empty()){
                return 0;
            }
            cout << "Input of my_mean (size:) " << v.size() << endl;
            //cout << "v.begin() " << v.begin() << endl;
            //cout << "v.begin() " << v.end() << endl;
            float sum = std::accumulate(v.begin(), v.end(), 0.0f);
            //return sum;
            cout << "mean: " << sum/v.size() << endl;
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

            //cout << "Last theoretical index " << *intersection_mask.end() << endl;
            //cout << "Last real index: " << *(intersection_mask.begin() + intersection_mask.size()-1) << endl;      //testing purposes 
            //cout << "Xintersection mask: " << xintersection_mask << endl; 
            //for(int v = 0; v< intersection_mask.size(); v++){
            //    cout << "Intersection mask  " << v << ": " << intersection_mask[v] << endl;
            //}
            //if(frame_counter==40){while(1);}     //testing purposes
            return xintersection_mask;
        }


        tuple<vector<vector<Point2f>>, xarray<int>> get_last_track_pair(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks){
            vector<Point2f> init_track1 = init_tracks[init_tracks.size()-1];
            vector<Point2f> init_track2 = init_tracks[init_tracks.size()-2];
            vector<int> init_mask1 = init_masks[init_masks.size()-1];
            vector<int> init_mask2 = init_masks[init_masks.size()-2];

            //auto xpair_mask = get_intersection_mask(init_mask1, init_mask2);

            auto xpair_mask = get_intersection_mask(init_mask2, init_mask1);
            assert(get_intersection_mask(init_mask1, init_mask2) == get_intersection_mask(init_mask2, init_mask1));
            auto xmask1 = adapt(init_mask1, {init_mask1.size()});
            auto xmask2 = adapt(init_mask2, {init_mask2.size()});

            auto xtrack1 = adapt(init_track1, {init_track1.size()});
            //cout << "TRACK 1: " << xtrack1 << endl;
            auto xtrack2 = adapt(init_track2, {init_track2.size()});
            //cout << "TRACK 2: " << xtrack2 << endl;

            //assert(xtrack1.size()==isin(xmask1, xpair_mask).size());
            //cout << "isin_mask1: " << isin(xmask1, xpair_mask);
            //cout << "isin_mask2: " << isin(xmask2, xpair_mask);
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