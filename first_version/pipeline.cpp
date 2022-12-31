#include <iostream>
#include <algorithm>
#include <string>
#include <numeric>
#define CERES_FOUND true
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/filesystem.hpp>
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

VideoCapture cap(path_to_vid);            //post visualization
//VideoCapture cap(0);                        //online visualization

class StructureFromMotion{
    private:
        vector<Point2f> prev_points, points, new_points;
        Mat frame, color, image, gray, prevGray;
        vector<Vec3f> Rs = {};
        vector <Vec3f> Ts = {};
        vector<Vec3f> points3d = {};
        vector<Vec3f> cloud = {};
        Matx33f K = {};
        const int frames_to_skip = 1;
        const LogTag TAG = LogTag("SfM", LOG_LEVEL_DEBUG);
        int frame_counter = 0;
        size_t i, j;
        const int MAX_COUNT = 500;
        //const int MAX_COUNT = 100;      //tcc param 
        const double closeness_threshold = 15;
        const int min_features = 100; //original:35
        bool needToInit = 1;
        //bool first_frame = 1;
        int start_index = 0;







    public:
        StructureFromMotion(){};


        void runSfM(){
            for(;;){
                feature_detector();

            }
        }

        void feature_detector(){
            cap >> frame;    
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
            cout << "INDEXES" << indexes << endl;
            
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

            cout << "INDEXES BEFORE TRACKING: " << indexes;
            auto xstatus = adapt(status, {status.size()});  
            indexes = filter(indexes, xstatus);
            cout << "INDEXES AFTER TRACKING: " << indexes;
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
            cout << adapt(closeness_table.shape()) << endl;
            cout << "CLOS TABLE " << closeness_table << endl;
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

            auto features = xt::hstack(xtuple(old_features, new_features));
            indexes = xt::hstack(xtuple(indexes, new_indexes));
            //cout << "FEATURES' SIZE: " << features.size() << endl;
            //cout << "FEATURES' DIMENSION: " << features.dimension() << endl;


            //cout << "OLD FEATURES: " << old_features << endl;
            //cout << "NEW FEATURES FILTERED: " << new_features << endl;
            //cout << "ALL FEATURES: " << features << endl;

            vector<Point2f> feats(features.begin(), features.end());
            points = feats;

            assert(points.size() ==indexes.size());
        }   






        double norm(Point2f a, Point2f b){
            return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
        }


        //THIS IS FOR ONLINE VISUALIZATION
        //Mat get_frames(){
        //    VideoCapture cap(0);
        //    if(i%(frames_to_skip+1) ==0){
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