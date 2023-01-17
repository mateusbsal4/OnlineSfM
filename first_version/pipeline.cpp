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
#include <typeinfo>
#include <algorithm>




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
//string path_to_vid =  "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/segundo_quarto.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_long/xadrez_cc.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_long/xadrez_cd.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_long/casa_cc.MOV";
string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/elef5.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_short/casa.MOV";




//THESE CRITERIA ARE SENSITIVE TO CHANGES
TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
Size subPixWinSize(10,10), winSize(31,31);            //example (standard OpenCV) criteria
//TermCriteria termcrit(1|2,30,0.003);
//Size subPixWinSize(10,10), winSize(15,15);           // tcc criteria

size_t initial_indexes_shape = 1;
xarray<int> indexes({initial_indexes_shape});

//size_t initial_cloud_shape = 1;
//xarray<float> cloud({initial_cloud_shape});


//size_t initial_features_shape = 0;
//xarray<double> features({initial_features_shape});



//VideoCapture cap(path_to_vid);            //post visualization
VideoCapture cap(0);                        //online visualization

class StructureFromMotion{
    private:
        vector<vector<Point2f>> tracks;
        vector<vector<int>> masks;
        vector<Point2f> track, features, prev_points, points, new_points;
        Mat frame, color, image, gray, prevGray;
        vector<int> frame_numbers, mask;
        int frame_number;
        vector<xarray<double>> Rs, Ts;       
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
        const int MAX_COUNT = 500;
        //const int MAX_COUNT = 100;      //tcc param
        const double closeness_threshold = 15;
        const int min_features = 100; //original:35
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







    public:
        StructureFromMotion(){};


        void runSfM(){
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
                    std::vector<vector<Point2f>> init_tracks = std::vector<vector<Point2f>> (tracks.begin(), tracks.begin()+init_reconstruction_frames);
                    std::vector<vector<int>> init_masks = std::vector<vector<int>> (masks.begin(), masks.begin()+init_reconstruction_frames);
                    std::vector<int> init_frame_numbers = std::vector<int> (frame_numbers.begin(), frame_numbers.begin()+init_reconstruction_frames);
                    assert(init_tracks.size() == init_masks.size());
                    init_reconstruction(init_tracks, init_masks);
                    continue;
                }
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
                get_new_features();
                needToInit = false;
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



        void get_new_features(){
            goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);   //example (standard OpenCV) params - block size (3) is too small! This selects too many features
            cornerSubPix(gray, new_points, subPixWinSize, Size(-1,-1), termcrit);
            //goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.5, 15, Mat(), 10,3, 0, 0.04);    //tcc params
//            if(!first_frame){
            if(!points.empty()){
                match_features();
            }
            else{
                points = new_points;
                indexes.resize({points.size()});
                indexes = arange(points.size());

            }
//            first_frame = 0;
        }

        void track_features(Mat mask, Mat image){
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, prev_points, points, status, err, winSize,
                                 3, termcrit, 0, 0.001);
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
            size_t n = prev_points.size();
            size_t m = new_points.size();
            vector<size_t> shape = {n, m};
            xarray<int> closeness_table(shape);
            for( i = 0; i < n; i++ ){
                for( j = 0; j < m; j++){
                    if(norm(prev_points[i],new_points[j]) <= closeness_threshold){
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
            auto old_features  = adapt(prev_points, {n});
            auto new_features = adapt(new_points, {m});



            //cout << "NEW FEATURES SIZE " << new_features.size() << endl;
            //cout << "NEW FEATURES DIMENSION " << new_features.dimension() << endl;
            //cout << "OLD FEATURES SIZE " << old_features.size() << endl;
            //cout << "FEATURES' DIMENSION: " << old_features.dimension() << endl;

            new_features = filter(new_features, new_points_mask);
            assert(ones_in_mask == new_features.size());
            auto new_indexes = arange(0, ones_in_mask)+start_index;


            //cout << "NEW FEATURES FILTERED: " << new_features << endl;
            //cout << "NEW FEATURES FILTERED SHAPE: " << adapt(new_features.shape()) << endl;
            //cout << "NEW FEATURES FILTERED SIZE " << new_features.size() << endl;
            //cout << "NEW FEATURES FILTERED DIMENSION " << new_features.dimension() << endl;
            //features = np.vstack((old_features, new_features))
            //indexes = np.concatenate((old_indexes, new_indexes))
            //vector<int> old_shape = {n, 1};
            //vector<size_t> new_shape = {ones_in_mask, 1};
            //old_features.resize(old_shape);
            //new_features.reshape(new_shape);
            //cout << "OLD FEATURES RESHAPED: " << old_features << endl;
            //cout << "NEW FEATURES RESHAPED: " << new_features << endl;
            //features.resize(old_features.size()+new_features.size());
            auto features = xt::hstack(xtuple(old_features, new_features));
            //auto features = xt::hstack(xtuple(old_features, new_features));
            indexes = xt::hstack(xtuple(indexes, new_indexes));
            //cout << "FEATURES' SIZE: " << features.size() << endl;
            //cout << "FEATURES' DIMENSION: " << features.dimension() << endl;

            //cout << "OLD FEATURES: " << old_features << endl;
            //cout << "NEW FEATURES FILTERED: " << new_features << endl;
            //cout << "ALL FEATURES: " << features << endl;
            //cout << "a feature: " << features(1) << endl;
            vector<Point2f> feats(features.begin(), features.end());
            points = feats;

            assert(points.size() ==indexes.size());
        }



        double norm(Point2f a, Point2f b){
            return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
        }


        void init_reconstruction(vector<vector<Point2f>> tracks, vector<vector<int>> masks){
            vector<vector<Point2f>> init_tracks;
            vector<vector<int>> init_masks;
            for(i = 0; i < tracks.size(); i++){
                init_tracks.push_back(tracks[i]);
                init_masks.push_back(masks[i]);

                if(cloud.empty()){
                    if(init_tracks.size() > 1 && init_tracks[0].size() >= 5){
                        five_pt_init(init_tracks, init_masks);
                    }
                    continue;
                }
            }


        }



        void five_pt_init(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks){
            if(init_tracks.size()>2){
                init_tracks = vector<vector<Point2f>> (init_tracks.end()-1, init_tracks.end());
                init_masks = vector<vector<int>> (init_masks.end()-1, init_masks.end());
            }
            Mat E, five_pt_mask, mask, R2, t2;
            xarray<double> R({3,3});
            xarray<double> T({3,1});
            //cout << "Dimension OF EMPTY R: " << R.dimension() << endl;
            //cout << "Dimension OF EMPTY T: " << T.dimension() << endl;
            //R = eye(3);
            //T = zeros<double>({3,1});
            //xarray<xarray<double>> Rs({1}); //this is not correct!!
            //xarray<xarray<double>> Ts({1});
            //xarray<xarray<double>> Rs({3,3,1}); //this is not correct!!
            //xarray<xarray<double>> Ts({3,1,1});
            //Rs += R;
            //Ts += T;
            //vector<xarray<double>> Rs = {eye(3)};
            //vector<xarray<double>> Ts = {zeros<double>({3,1})};
            //xarray<xarray<double>> Rs = {eye(3)};
            //xarray<xarray<double>> Ts = {zeros<double>({3,1})};
            vector<vector<Point2f>> track_pair;
            xarray<int> pair_mask;
            tie(track_pair, pair_mask)  = get_last_track_pair(init_tracks, init_masks);
            E = findEssentialMat(track_pair[0], track_pair[1], K,  RANSAC, ransac_probability, essential_mat_threshold, five_pt_mask);         //error was here
            five_pt_mask.copyTo(mask);
            //cout << "Input mask = " << mask << endl;

            recoverPose(E, track_pair[0], track_pair[1], K, R2, t2, distance_thresh, five_pt_mask);

            //cout << "Output mask = " << five_pt_mask << endl;


            cout << "Mat R = " << endl << " " << R2 << endl << endl;
            cout << "Mat T = " << endl << " " << t2 << endl << endl;

            R = mat_to_xarray(R2);
            T = mat_to_xarray(t2);
            cout << "xarray R: " << R << endl;
            cout << "xarray T: " << T << endl;
            tie(R, T) = invert_reference_frame(R,T);
            cout << "Dimension OF FILLED R: " << R.dimension() << endl;
            cout << "Dimension OF FILLED T: " << T.dimension() << endl;
            cout << "Dimension of first R: " << Rs[0].dimension() << endl;
            cout << "Dimension of first T: " << Ts[0].dimension() << endl;

            cout << "Transposed R: " << R << endl;
            cout << "Linalg T: " << T << endl;
            assert(Rs.size()-1 ==0);
            assert(Ts.size()-1 ==0);
            tie(R, T) = compose_rts(R, T, Rs[0], Ts[0]);
            //assert(R.dimension()==2 && T.dimension()==2 && Rs(0).dimension()==2 && Ts(0).dimension()==2);
            vector<Point3f> points3d;
            if(R.dimension()==2 && T.dimension()==2 && Rs[0].dimension()==2 && Ts[0].dimension()==2){
                points3d = triangulate(transpose(Rs[0]), -linalg::dot(transpose(Rs[0]), Ts[0]), transpose(R), -linalg::dot(transpose(R),T), track_pair);                                              //reconverting motion matrices to relative frame coordinate system is required by the OpenCV function
            }
            points_to_cloud(points3d, pair_mask); 
            Rs.push_back(R);
            Ts.push_back(T);

        }

        void points_to_cloud(vector<Point3f> points3d, xarray<int> pair_mask){
            cout << "Max track_mask: " << amax(pair_mask)() +1 << endl;
            auto points_3d = adapt(points3d, {points3d.size()});
            cout << "XPointstocloud Dimension: " << points_3d.dimension() << endl;
            cout << "XPointstocloud Size: " << points_3d.size() << endl;
            cout << "XPointstocloud SHape: " <<  adapt(points_3d.shape()) << endl;
            //auto xcloud = full_like(points_3d, {0.0,0.0,0.0});
            //size_t szxcloud = amax(pair_mask)() +1;
            xarray<Point3f> xcloud = {{0.0,0.0,0.0}};
            cout << "Xcloud Dimension: " << xcloud.dimension() << endl;
            cout << "Xcloud Size: " << xcloud.size() << endl;
            cout << "Xcloud SHape: " <<  adapt(xcloud.shape()) << endl;
            xcloud.resize({amax(pair_mask)() +1});
            cout << "Xcloud Dimension: " << xcloud.dimension() << endl;
            cout << "Xcloud Size: " << xcloud.size() << endl;
            cout << "Xcloud SHape: " <<  adapt(xcloud.shape()) << endl;
            cout << "Xcloud 2nd el: " << xcloud(1) << endl;
            xcloud = filter(xcloud, pair_mask);
            xcloud = points_3d;
            assert(xcloud.size() == points_3d.size());
            assert(adapt(xcloud.shape()) == adapt(points_3d.shape()));            
            assert(xcloud.dimension() == points_3d.dimension());
            //xcloud.filter(pair_mask) = points_3d;
            cout << "Xcloud:" << xcloud;
            cout << "Indexes:" << pair_mask;
            vector<Point3f> points_cloud(xcloud.begin(), xcloud.end());
            cloud = points_cloud;
        }


        vector<Point3f> triangulate(xarray<double> R1, xarray<double> T1, xarray<double> R2, xarray<double> T2, vector<vector<Point2f>> track_pair){
            xarray<double> P1({3,4});
            xarray<double> P2({3,4});
            xarray<double> xK = mat_to_xarray(K);
            cout << "K in xarray = " << xK << endl;
            P1 = linalg::dot(xK, hstack(xtuple(R1, T1)));
            P2 = linalg::dot(xK, hstack(xtuple(R2, T2)));
            cout << "R1 = " << R1 << endl;
            cout << "T1 = " << T1 << endl;
            cout << "[R1|T1] = " << hstack(xtuple(R1, T1));
            cout << "R2 = " << R2 << endl;
            cout << "T2 = " << T2 << endl;
            cout << "[R2|T2] = " << hstack(xtuple(R2, T2));
            cout << "Proj matrix 1: " << P1 << endl;
            cout << "Proj matrix 2: " << P2 << endl;
            Mat P_1, P_2, points4d;
            P_1 = xarray_to_mat_elementwise(P1);
            P_2 = xarray_to_mat_elementwise(P2);
            cout << "X Proj matrix1: " << P_1 << endl;
            cout << "X Proj matrix2: " << P_2 << endl;
            triangulatePoints(P_1, P_2, track_pair[0], track_pair[1], points4d);
            printf("Matrix: %s %dx%d \n", type2str( points4d.type() ).c_str(), points4d.rows, points4d.cols );
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
            vector<Point3f> points3d;
            convertPointsFromHomogeneous(pts4d, points3d);
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


        tuple<xarray<double>, xarray<double>> invert_reference_frame(xarray<double> R, xarray<double> T){
            return make_tuple(transpose(R), linalg::dot(transpose(R), -T));                                   //expressing motion matrices between current and previous frame
        }                                                                                                      // in global (first camera) coordinate system

        tuple<xarray<double>, xarray<double>> compose_rts(xarray<double> R, xarray<double> T, xarray<double> prev_R, xarray<double> prev_T){
            return make_tuple(linalg::dot(prev_R,R), prev_T + linalg::dot(prev_R, T));                      //(og pipeline: compose_rts - this expresses motion from the current frame in respect to the first)
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


        tuple<vector<vector<Point2f>>, xarray<int>> get_last_track_pair(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks){
            vector<Point2f> init_track1 = init_tracks[init_tracks.size()-1];
            vector<Point2f> init_track2 = init_tracks[init_tracks.size()-2];
            vector<int> init_mask1 = init_masks[init_masks.size()-1];
            vector<int> init_mask2 = init_masks[init_masks.size()-2];
            vector<int> pair_mask;

            for(i = 0; i<init_mask1.size(); i++){
                for(j=0; j<init_mask2.size(); j++){
                    if(init_mask1[i]==init_mask2[j]){
                        pair_mask.push_back(init_mask1[i]);
                    }
                }
            }
            sort(pair_mask.begin(), pair_mask.end());
            auto xpair_mask = adapt(pair_mask, {pair_mask.size()});
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


            vector<vector<Point2f>> track_pair{track_pair0, track_pair1};

            //vector<vector<Point2f>> track_pair;
            //track_pair.push_back(track_pair0);
            //track_pair.push_back(track_pair1);


            return make_tuple(track_pair, xpair_mask);


        }












        //THIS IS FOR ONLINE VISUALIZATION
        //Mat get_frames(){
        //    VideoCapture cap(0);
        //    if(i%(frames_to_ski   p+1) ==0){
        //            //CV_LOG_INFO(&TAG, "Reading frames");
        //            //cap.read(frame);
        //            cap >> frame;
        //            //imshow("Frame", frame);
        //            //waitKey(1);
        //            return frame;
        //    }
        //    //i++;
        //}

        // THIS IS FOR POST VISUALIZATION
        //Mat get_frames_video(){
        //    //frame_counter ++;    //needs solving!! counter is NOT working
        //    if(frame_counter%(frames_to_skip+1) ==0){
        //            cap >> frame;
        //            return frame;
//
        //    }
        //}




};


int main(){
    StructureFromMotion sfm;
    sfm.runSfM();
    return 0;
}