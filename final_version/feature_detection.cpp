#include "definitions.h"
#include "pipeline.h"

/*      Feature detection criteria      */
//TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,50,0.01);
TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,50,0.01);
Size subPixWinSize(10,10), winSize(31,31);            //example (standard OpenCV) criteria
//TermCriteria termcrit(3,30,0.003);
//Size subPixWinSize(10,10), winSize(15,15);           

int initial_indexes_shape = 1;   
xarray<int> indexes_2d({initial_indexes_shape});

tuple<int, vector<Point2f>, vector<int>> StructureFromMotion::feature_detector(){
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

void StructureFromMotion::get_new_features(Mat mask, Mat image){
    //goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);   //example (standard OpenCV) params - block size (3) is too small! This selects too many features
    //goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.01, 10, Mat(), 7, 3, 0, 0.04);   
    goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.01, 10, Mat(), 7, 3, 0, 0.04);   
    cornerSubPix(gray, new_points, subPixWinSize, Size(-1,-1), termcrit);
    //goodFeaturesToTrack(gray, new_points, MAX_COUNT, 0.5, 15, Mat(), 11,3, 0, 0.04);    //tcc params
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

void StructureFromMotion::track_features(Mat mask, Mat image){
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

void StructureFromMotion::match_features(){
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
    indexes_2d = xt::hstack(xtuple(indexes_2d, new_indexes));
    vector<Point2f> feats(features.begin(), features.end());            
    points = feats;
    assert(points.size() ==indexes_2d.size());
}