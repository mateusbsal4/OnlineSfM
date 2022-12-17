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
#include <iostream>
#include <ctype.h>
#include <tuple>
#include <cmath>
#include <Eigen/Dense>
using namespace std;
using namespace cv;
using namespace cv::utils::logging;
using namespace Eigen;

bool needToInit = true;
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_long/casa_cc.MOV";
string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/elef5.MOV";
TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
Size subPixWinSize(10,10), winSize(31,31);
Mat color, gray,image, prevGray;
vector<Point2f> points[2];
vector<unsigned int> indexes;
VideoCapture cap(path_to_vid);
Mat frame;

class StructureFromMotion{
    private:
        vector<Vec3f> Rs = {};
        vector <Vec3f> Ts = {};
        vector<Vec3f> points3d = {};
        vector<Vec3f> cloud = {};
        Matx33f K = {};
        const int frames_to_skip = 1;
        const LogTag TAG = LogTag("SfM", LOG_LEVEL_DEBUG);
        size_t frame_counter =0;
        size_t i, j;
        const int MAX_COUNT = 500;
        const double closeness_threshold = 15;
        const int min_features = 35; 








    public:
        StructureFromMotion(){};


        void runSfM(){
            for(;;){
                feature_detector();
            }
        }

        void feature_detector(){    
            CV_LOG_INFO(&TAG, "Detecting Features");
            //color = get_frames();
            color = get_frames_video();
            Mat mask = Mat::zeros(color.size(), color.type());
            Mat final_img;
            color.copyTo(image);
            cvtColor(image, gray, COLOR_BGR2GRAY);
            if( needToInit )
            {
                get_new_features();
                needToInit = false;
            }
            //else if( !points[0].empty() ){
            //    track_features(mask);
            //}
            if( !points[0].empty() ){
                track_features(mask);
            }
            //needToInit = false;       //[mbs:221129]: testing match_features addition
            add(image, mask, final_img);
            imshow("LK Tracker", final_img);
            waitKey(10);
            std::swap(points[1], points[0]);
            cv::swap(prevGray, gray);
            if(points[0].size() < min_features){
                needToInit = true;
            }
            
        }



        void get_new_features(){
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            //for(int i =0; i < points[1].size(); i++){
            //    indexes.push_back(i);
            //    cout<< indexes[i] << endl; 
            //}
            match_features();
        

        }

        void track_features(Mat mask){
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            for( i = j = 0; i < points[1].size(); i++ )
            {
                if( !status[i] )
                    continue;
                points[1][j++] = points[1][i];
                line(mask, points[0][i], points[1][i], Scalar(255,0,0), 2);
                circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
            }
            points[1].resize(j);
        }


        void match_features(){
            int n = points[0].size();
            int m = points[1].size();
            MatrixXf closeness_table(n,m);
//            size_t k, l;
            for( i = 0; i < n; i++ ){
                for( j = 0; j < m; j++){
                    if(norm(points[0][i],points[1][j]) <= closeness_threshold){
                        closeness_table(i,j)=1;
                    }
                    else{
                        closeness_table(i,j) =0;
                    }
                }
            }
            VectorXf new_points_mask(m);
            new_points_mask = closeness_table.colwise().sum();   
            MatrixXf new_features(m,2);
            for(i = 0; i< m; i++){
                if(new_points_mask(i)){
                    float* p1 = &points[1][i].x;
                    float* p2 = &points[1][i].y;  
                    new_features(i,0) = *p1;    
                    new_features(i,1) = *p2;    
                }
                //  else{
                //      removeRow(new_features, i);
                //  }
                //cout << "x coordinate of the point:" << new_features(i,0) << endl;
            }
            cout << new_features << endl;
            cout << "SIZE OF " << new_features.rows() << endl;
            cout << "TOTAL OF " << m << endl;
            //points[1] = points[1][new_points_mask];
            
            //for (k=0; k<m; k++){
            //    if 
            //}
            //for( k = 0; k< m; k++){
            //    cout << closeness_table.col(k);
            //    new_points_mask(k) = closeness_table.col(k).sum();
            //}

        }   



        //array get_close_points_table(int n, int m){
        //    int n = points[0].size();
        //    int m = points[1].size();
        //    int table[n][m];
        //    size_t k, l;
        //    for( k = 0; k < n; k++ ){
        //        for( l = 0; l < m; l++){
        //            if(norm(points[0][k],points[1][l]) <= closeness_threshold){
        //                table[k][l]==1;
        //            }
        //            else{
        //                table[k][l]==0;
        //            }
        //        }
        //    }
        //    return table;
        //}

        //void removeRow(Eigen::MatrixXf& matrix, unsigned int rowToRemove)
        //{
        //    unsigned int numRows = matrix.rows()-1;
        //    unsigned int numCols = matrix.cols();
//
        //    if( rowToRemove < numRows )
        //        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);
//
        //    matrix.conservativeResize(numRows,numCols);
        //}
//
//
        //void removeColumn(Eigen::MatrixXf& matrix, unsigned int colToRemove)
        //{
        //    unsigned int numRows = matrix.rows();
        //    unsigned int numCols = matrix.cols()-1;
//
        //    if( colToRemove < numCols )
        //        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);
//
        //    matrix.conservativeResize(numRows,numCols);
        //}
//
        double norm(Point2f a, Point2f b){
            return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
        }


        //THIS IS FOR ONLINE VISUALIZATION
        //Mat get_frames(){
        //    VideoCapture cap(0);
        //    Mat frame;  
        //    if(i%(frames_to_skip+1) ==0){
        //            //CV_LOG_INFO(&TAG, "Reading frames");
        //            //cap.read(frame);
        //            cap >> frame;
        //            //imshow("Frame", frame);
        //            //waitKey(1);
        //            return frame;
        //    }
        //    i++;
        //}

        // THIS IS FOR POST VISUALIZATION
        Mat get_frames_video(){
                if(frame_counter%(frames_to_skip+1) ==0){
                        cap >> frame;
                        return frame;
                }
                frame_counter++;
        }




};


int main(){
    StructureFromMotion sfm;
    sfm.runSfM();

}