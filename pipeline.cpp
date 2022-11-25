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
using namespace std;
using namespace cv;
using namespace cv::utils::logging;

bool needToInit = true;
string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_long/casa_cc.MOV";
//string path_to_vid = "/home/mateus/IC/tcc_sfm-master_2/tcc_sfm-master/datasets/pasta_medium/elef5.MOV";
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
        int i = 0;
        const int MAX_COUNT = 500;





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
            }
            else if( !points[0].empty() ){
                track_features(mask);
            }
            needToInit = false;
            add(image, mask, final_img);
            imshow("LK Tracker", final_img);
            waitKey(10);
            std::swap(points[1], points[0]);
            cv::swap(prevGray, gray);
        }


        void get_new_features(){
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
        }




        void track_features(Mat mask){
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i, k;
            for( i = k = 0; i < points[1].size(); i++ )
            {
                if( !status[i] )
                    continue;
                points[1][k++] = points[1][i];
                line(mask, points[0][i], points[1][i], Scalar(255,0,0), 2);
                circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
            }
            points[1].resize(k);
        }





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


        Mat get_frames_video(){
            if(i%(frames_to_skip+1) ==0){
                    cap >> frame;
                    return frame;
            }
            i++;
        }




};


int main(){
    StructureFromMotion sfm;
    sfm.runSfM();

}