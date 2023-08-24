#include "definitions.h"
#include "pipeline.h"

xarray<int> StructureFromMotion::get_nan_index_mask(vector<Point3f> cloud){
    xarray<bool> nan_bool_mask = get_nan_bool_mask(cloud);
    return filter(arange(cloud.size()), nan_bool_mask);
}

xarray<int> StructureFromMotion::get_not_nan_index_mask(vector<Point3f> cloud){
    xarray<bool> not_nan_bool_mask = !get_nan_bool_mask(cloud);
    return filter(arange(cloud.size()), not_nan_bool_mask);
}

xarray<bool> StructureFromMotion::get_nan_bool_mask(vector<Point3f> cloud){
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

vector<Point3f> StructureFromMotion::points_to_cloud(vector<Point3f> points3d, xarray<int> pair_mask){
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

vector<Point3f> StructureFromMotion::add_points_to_cloud(vector<Point3f> points_3d, xarray<int> indexes, vector<Point3f> cloud){   //NOTE: this function only adds new points to the cloud, it does not replace any point that has already been triangulated.
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

xarray<Point3f> StructureFromMotion::double_filter(xarray<Point3f> first_vec, xarray<Point3f> second_vec, xarray<int> mask1, xarray<int> mask2){
    for(const int & index: mask1){
        first_vec(index) = second_vec(index);
    }
    return first_vec;
} 

xarray<Point3f> StructureFromMotion::all_nan(xarray<Point3f> vector_3d){
    for(i =0; i<vector_3d.size(); i++){
        vector_3d(i).x = numeric_limits<float>::quiet_NaN();            
        vector_3d(i).y = numeric_limits<float>::quiet_NaN();
        vector_3d(i).z = numeric_limits<float>::quiet_NaN();             
    }
    return vector_3d;
}

xarray<float> StructureFromMotion::all_nan_list(xarray<float> list){
    for(i =0; i<list.size(); i++){
        list(i)= numeric_limits<float>::quiet_NaN();                    
    }
    return list;
}

tuple<Mat, Mat> StructureFromMotion::invert_reference_frame(Mat R, Mat T){
    if(R.empty()){
        return make_tuple(T, R);
    }
    return make_tuple(R.t(), -(R.t())*T);                                   //expressing motion matrices between current and previous frame
}                                                                           // in global (first camera) coordinate system

tuple<Mat, Mat> StructureFromMotion::compose_rts(Mat R, Mat T, Mat prev_R, Mat prev_T){
    return make_tuple(prev_R*R, prev_T + prev_R*T);                      //this expresses motion from the current frame in respect to the first
}
        
Mat StructureFromMotion::xarray_to_mat_elementwise(xt::xarray<double> xarr)
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

xarray<double> StructureFromMotion::mat_to_xarray(cv::Mat mat)
{
    xarray<double> res = xt::adapt((double*) mat.data, mat.cols * mat.rows, xt::no_ownership(), std::vector<std::size_t> {mat.rows, mat.cols});
    return res;
}

float StructureFromMotion::my_mean(std::vector<float> const& v)
{
    if(v.empty()){
        return 0;
    }
    float sum = std::accumulate(v.begin(), v.end(), 0.0f);
    return sum / v.size();
}

xarray<int> StructureFromMotion::get_intersection_mask(vector<int> mask1, vector<int> mask2){
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

tuple<vector<vector<Point2f>>, xarray<int>> StructureFromMotion::get_last_track_pair(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks){
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

vector<int> StructureFromMotion::full_of_ints(size_t size, int data){
    vector<int> vec;
    for(size_t c = 0; c<size; c++){
        vec.push_back(data);
    }
    return vec;
}

double StructureFromMotion::norm(Point2f &a, Point2f &b){
    return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}