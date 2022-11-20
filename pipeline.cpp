#include <iostream>
#include <algorithm>
#include <string>
#include <numeric>
#define CERES_FOUND true
#include <opencv2/opencv.hpp>
//#include <opencv2/sfm.hpp>
//#include <opencv2/viz.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/filesystem.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <boost/filesystem.hpp>
//#include <boost/graph/graph_traits.hpp>
//#include <boost/graph/adjacency_list.hpp>
//#include <boost/graph/connected_components.hpp>
//#include <boost/graph/graphviz.hpp>
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

class StructureFromMotion{
    private:
        vector<Mat> features = {};
        vector<Vec3f> Rs = {};
        vector <Vec3f> Ts = {};
        vector<Vec3f> points3d = {};
        vector<Vec3f> cloud = {};
        Matx33f K = {};
        const int frames_to_skip = 1;
        const LogTag TAG = LogTag("SfM", LOG_LEVEL_DEBUG);
        int i = 0;
        const int MAX_COUNT = 500;
        //const double quality_level = 0.01;
        //const double min_distance = 10;
        //const int block_size = 10;
        //const int gradient_size = 10;
        //const double k  =0.04;









    public:
        StructureFromMotion(){};


        void runSfM(){
            feature_detector();
        }

        void feature_detector(){
            CV_LOG_INFO(&TAG, "Detecting Features");
            TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
            Size subPixWinSize(10,10), winSize(31,31);
            Mat color, gray,image, prevGray;
            vector<Point2f> points[2];
            for(;;){
                color = get_frames();
                color.copyTo(image);
                cvtColor(image, gray, COLOR_BGR2GRAY);
                if( needToInit )
                {
                    cout<< "ENTERING?" << endl;
                    goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
                    cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
                }
                else if( !points[0].empty() ){
                    cout<< "ENTERING 2??"<< endl;
                    vector<uchar> status;
                    vector<float> err;
                    if(prevGray.empty())
                        gray.copyTo(prevGray);
                    calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                         3, termcrit, 0, 0.001);
                    size_t i, k;
                    for( i = k = 0; i < points[1].size(); i++ )
                    {
                        cout<< "ENTERING 2?????"<< endl;
                        if( !status[i] )
                            continue;
                        points[1][k++] = points[1][i];
                        cout << "POINTS[1][I]" << points[1][i] << endl;
                        circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
                    }
                    points[1].resize(k);
                }
                needToInit = false;
                imshow("LK Demo", image);
                waitKey(10);
                std::swap(points[1], points[0]);
                cv::swap(prevGray, gray);
                cout<< "POINTS[0]" << points[0] << endl;
            }
    


            //CV_LOG_INFO(&TAG, "Detecting Features");
            //for(auto [frame, gray_frame]: get_frames()){
            //    cout << "TOWNN";
//
            //}
//


        }

        Mat get_frames(){
            VideoCapture cap(0);
            Mat frame;  
            if(i%(frames_to_skip+1) ==0){
                    //CV_LOG_INFO(&TAG, "Reading frames");
                    cap.read(frame);
                    //imshow("Frame", frame);
                    //waitKey(1);
                    return frame;
            }
            i++;
        }

        







};


int main(){
    StructureFromMotion sfm;
    sfm.runSfM();

}