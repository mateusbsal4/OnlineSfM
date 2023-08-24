#ifndef PIPELINE_H
#define PIPELINE_H
#include "definitions.h"

int main(int argc, char ** argv);

class StructureFromMotion{
    public:
        /*  PIPELINE    */
        StructureFromMotion();
        int runSfM(VideoCapture cap);
        /*  ------------ */
        /* FEATURE_DETECTION */
        tuple<int, vector<Point2f>, vector<int>> feature_detector();
        void get_new_features(Mat mask, Mat image);
        void track_features(Mat mask, Mat image);
        void match_features();
        /*  ------------ */
        /*  RECONSTRUCTION  */
        bool check_end(VideoCapture cap);
        tuple<vector<Mat>, vector<Mat>, vector<Point3f>> init_reconstruction(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud);
        tuple<vector<Mat>, vector<Mat>, vector<Point3f>> reconstruct(vector<vector<Point2f>> tracks, vector<vector<int>> masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud);
        tuple<vector<Mat>, vector<Mat>, vector<Point3f>> five_pt_init(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud);
        tuple<Mat, Mat> five_pt(vector<vector<Point2f>> track_pair, xarray<int> pair_mask, Mat prev_R, Mat prev_T);
        tuple<Mat, Mat> solve_pnp(vector<Point2f> track, vector<int> mask, Mat R_est, Mat T_est, vector<Point3f> cloud);
        tuple<Mat, Mat> solve_pnp_(vector<Point2f> track_slice, vector<int> track_mask, cv::SolvePnPMethod method, Mat R, Mat T, vector<Point3f> cloud);
        vector<Point3f> triangulate(Mat R1, Mat T1, Mat R2, Mat T2, vector<vector<Point2f>> track_pair);     
        tuple<Mat, Mat, vector<Point3f>, xarray<int>> calculate_projection(Mat prev_R, Mat prev_T, vector<vector<Point2f>> tracks, vector<vector<int>> masks, vector<Point3f> cloud);
        float calculate_init_error(vector<vector<Point2f>> error_calc_tracks, vector<vector<int>> error_calc_masks, vector<Point3f> pt_cloud);
        float calculate_reconstruction_error(vector<Mat> ec_Rs, vector<Mat> ec_Ts, vector<vector<Point2f>> ec_tracks, vector<vector<int>> ec_masks, vector<Point3f> pt_cloud);  
        /*  ------------ */     
        /*     UTILS     */
        xarray<int> get_nan_index_mask(vector<Point3f> cloud);
        xarray<int> get_not_nan_index_mask(vector<Point3f> cloud);
        xarray<bool> get_nan_bool_mask(vector<Point3f> cloud);
        vector<Point3f> points_to_cloud(vector<Point3f> points3d, xarray<int> pair_mask);
        vector<Point3f> add_points_to_cloud(vector<Point3f> points_3d, xarray<int> indexes, vector<Point3f> cloud);    
        xarray<Point3f> double_filter(xarray<Point3f> first_vec, xarray<Point3f> second_vec, xarray<int> mask1, xarray<int> mask2 = xt::xarray<int>{});
        xarray<Point3f> all_nan(xarray<Point3f> vector_3d);
        xarray<float> all_nan_list(xarray<float> list);
        tuple<Mat, Mat> invert_reference_frame(Mat R, Mat T);
        tuple<Mat, Mat> compose_rts(Mat R, Mat T, Mat prev_R, Mat prev_T);
        Mat xarray_to_mat_elementwise(xt::xarray<double> xarr);
        xarray<double> mat_to_xarray(cv::Mat mat);
        float my_mean(std::vector<float> const& v);
        xarray<int> get_intersection_mask(vector<int> mask1, vector<int> mask2);
        tuple<vector<vector<Point2f>>, xarray<int>> get_last_track_pair(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks);
        vector<int> full_of_ints(size_t size, int data);
        double norm(Point2f &a, Point2f &b);
        /*  ------------ */     
        /*     BUNDLE_ADJUSTMENT     */
        tuple<vector<vector<double>>, vector<Point3f>, vector<Point2f>, vector<int>, vector<int>> prepare_optimization_input(vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud, vector<vector<Point2f>> tracks, vector<vector<int>> masks);
        void run_ba(vector<Mat>& Rs, vector<Mat>& Ts, vector<Point3f>& cloud, vector<vector<Point2f>>& tracks, vector<vector<int>>& masks);
        /*  ------------ */     

    private:
        vector<vector<Point2f>> tracks;
        vector<vector<int>> masks;
        vector<Point2f> track, features, prev_points, points, new_points;
        Mat frame, color, image, gray, prevGray;
        vector<int> frame_numbers, mask; 
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
        const int MAX_COUNT = 300; //100 
        const double closeness_threshold = 15;
        const int min_features = 150; //100 or 150 are also options
        bool needToInit = 1;
        int start_index = 0;
        float error_threshold = 8;  //empirically best value is ~1.5, in some cases 4, 3.2 also work. Depends on the example
        int init_reconstruction_frames = 5;
        int error_calculation_frames = 5;
        bool is_init = 1;
        double ransac_probability = 0.999999;
        double essential_mat_threshold = 5;
        double distance_thresh = 500;
        bool use_epnp = 1;
        bool use_iterative_pnp = 1;
        int min_number_of_points = 5;
        bool use_five_pt_algorithm = 0;
        bool use_solve_pnp = 1;
        int dropped_tracks = 0;
        int ba_window = 100;
        int adjust_path = 0;

};

#endif  //PIPELINE_H