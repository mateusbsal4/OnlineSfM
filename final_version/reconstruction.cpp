#include "definitions.h"
#include "pipeline.h"

bool StructureFromMotion::check_end(VideoCapture cap){
    cap >> frame;
    return (frame.empty())?1:0;
}

tuple<vector<Mat>, vector<Mat>, vector<Point3f>> StructureFromMotion::init_reconstruction(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud){
    Mat R, T;
    vector<Point3f> pts_3d;
    xarray<int> indexes;
    for(initial_counter = 0; initial_counter < init_reconstruction_frames; initial_counter++){      
        vector<vector<Point2f>> init_tracks_sliced  = std::vector<vector<Point2f>> (init_tracks.begin(), init_tracks.begin()+initial_counter+1);                     
        vector<vector<int>> init_masks_sliced = std::vector<vector<int>> (init_masks.begin(), init_masks.begin()+initial_counter+1);                       
        if(cloud.empty()){
            if(init_masks_sliced.size() > 1 && init_tracks_sliced[0].size() >= 5){
                tie(Rs, Ts, cloud) = five_pt_init(init_tracks_sliced, init_masks_sliced, Rs, Ts, cloud);
                assert(Rs.size() == 2);
            }   
            continue;
        }
        tie(R, T, pts_3d, indexes) = calculate_projection(Rs[Rs.size()-1], Ts[Ts.size()-1], init_tracks_sliced, init_masks_sliced, cloud);
        if(!pts_3d.empty()){
            cloud = add_points_to_cloud(pts_3d, indexes, cloud); 
        }
        Rs.push_back(R);
        Ts.push_back(T);
    }
    return make_tuple(Rs, Ts, cloud);  
}

tuple<vector<Mat>, vector<Mat>, vector<Point3f>> StructureFromMotion::reconstruct(vector<vector<Point2f>> tracks, vector<vector<int>> masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud){
    Mat R, T; 
    vector<Point3f> new_pts;
    xarray<int> new_pt_indexes;
    tie(R, T, new_pts, new_pt_indexes) = calculate_projection(Rs[Rs.size() - 1], Ts[Ts.size() - 1], tracks, masks, cloud);
    if(!R.empty() && !T.empty()){
        if(!new_pts.empty()){
            cloud = add_points_to_cloud(new_pts, new_pt_indexes, cloud);
        }
        Rs.push_back(R);
        Ts.push_back(T);
    }
    return make_tuple(Rs, Ts, cloud);
}

tuple<vector<Mat>, vector<Mat>, vector<Point3f>> StructureFromMotion::five_pt_init(vector<vector<Point2f>> init_tracks, vector<vector<int>> init_masks, vector<Mat> Rs, vector<Mat> Ts, vector<Point3f> cloud){
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
    points3d = triangulate(Rs[0].t(), -(Rs[0].t())*Ts[0], R.t(), -(R.t())*T, track_pair);                                              //reconverting motion matrices to relative frame coordinate system is required by the OpenCV function
    assert(points3d.size() == pair_mask.size());
    cloud = points_to_cloud(points3d, pair_mask); 
    Rs.push_back(R);
    Ts.push_back(T);
    return make_tuple(Rs, Ts, cloud);
}

tuple<Mat, Mat> StructureFromMotion::five_pt(vector<vector<Point2f>> track_pair, xarray<int> pair_mask, Mat prev_R, Mat prev_T){
    Mat E, five_pt_mask, mask, R, T; 
    E = findEssentialMat(track_pair[0], track_pair[1], K,  RANSAC, ransac_probability, essential_mat_threshold, five_pt_mask);       
    five_pt_mask.copyTo(mask);
    recoverPose(E, track_pair[0], track_pair[1], K, R, T, distance_thresh, five_pt_mask);
    tie(R, T) = invert_reference_frame(R,T);
    tie(R, T) = compose_rts(R, T, prev_R, prev_T);
    return make_tuple(R,T);
}

tuple<Mat, Mat> StructureFromMotion::solve_pnp(vector<Point2f> track, vector<int> mask, Mat R_est, Mat T_est, vector<Point3f> cloud){
    Mat R, T;
    if(use_epnp){
        tie(R, T) = solve_pnp_(track, mask, SOLVEPNP_EPNP, R_est, T_est, cloud);
        R_est = R;
        T_est = T;
    }   
    if(use_iterative_pnp){
        tie(R, T) = solve_pnp_(track, mask, SOLVEPNP_ITERATIVE, R_est, T_est, cloud);
    }
    return make_tuple(R, T);
}

tuple<Mat, Mat> StructureFromMotion::solve_pnp_(vector<Point2f> track_slice, vector<int> track_mask, cv::SolvePnPMethod method, Mat R, Mat T, vector<Point3f> cloud){
   bool use_extrinsic_guess = (!R.empty() && !T.empty()) ? 1 : 0;
   xarray<int> xcloud_mask = get_not_nan_index_mask(cloud); 
   vector<int> cloud_mask(xcloud_mask.begin(), xcloud_mask.end());
   xarray<int> intersection_mask = get_intersection_mask(cloud_mask, track_mask);
   xarray<int> xtrack_mask = adapt(track_mask, {track_mask.size()});
   xarray<bool> track_bool_mask = isin(xtrack_mask, intersection_mask);
   if(intersection_mask.size()<min_number_of_points){
        return make_tuple(R,T);
   }
   tie(R, T) = invert_reference_frame(R,T);
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
   assert(pts_cloud.size() == img_points.size());
   solvePnP(pts_cloud, img_points, K, distCoeffs, rvec, T, use_extrinsic_guess, method);
   Rodrigues(rvec, R);
   tie(R, T) = invert_reference_frame(R, T);
   return make_tuple(R, T);
}

tuple<Mat, Mat, vector<Point3f>, xarray<int>> StructureFromMotion::calculate_projection(Mat prev_R, Mat prev_T, vector<vector<Point2f>> tracks, vector<vector<int>> masks, vector<Point3f> cloud){
    int threshold = 40;
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
    if(!prev_T.empty()){
        if((abs(T.at<double>(0,0)) > abs(threshold*prev_T.at<double>(0,0)) || abs(T.at<double>(1,0)) > abs(threshold*prev_T.at<double>(1,0)) || abs(T.at<double>(2,0))> abs(threshold*prev_T.at<double>(2,0)))){
            Mat empty_R, empty_T;
            vector<Point3f> empty_points;
            return make_tuple(empty_R, empty_T, empty_points, pair_mask);
        }
    }
    vector<Point3f> points_3d = triangulate(prev_R.t(), -(prev_R.t())*prev_T, R.t(), -(R.t())*T, track_pair);           //careful here!! Cv operations won't work with empty matrices
    return make_tuple(R, T, points_3d, pair_mask);  
}

float StructureFromMotion::calculate_init_error(vector<vector<Point2f>> error_calc_tracks, vector<vector<int>> error_calc_masks, vector<Point3f> pt_cloud){
    size_t c;
    vector<Mat> error_calc_Rs;
    vector<Mat> error_calc_Ts;
    Mat R, T, empty_R, empty_T;
    assert(error_calc_masks.size() == error_calc_tracks.size());
    for(c =0; c<error_calc_masks.size(); c++){
        tie(R, T) = solve_pnp(error_calc_tracks[c], error_calc_masks[c], empty_R, empty_T, pt_cloud);
        error_calc_Rs.push_back(R);
        error_calc_Ts.push_back(T);             
    }
    return calculate_reconstruction_error(error_calc_Rs, error_calc_Ts, error_calc_tracks, error_calc_masks, pt_cloud); //xadrez_cc memory leakage occurs here
}

float StructureFromMotion::calculate_reconstruction_error(vector<Mat> ec_Rs, vector<Mat> ec_Ts, vector<vector<Point2f>> ec_tracks, vector<vector<int>> ec_masks, vector<Point3f> pt_cloud){
    size_t c;
    Mat R, T, R_cam, r_cam_vec, T_cam, distCoeffs;
    vector<Point2f> original_track, projected_track;
    vector<int> track_mask;
    xarray<int> xtrack_mask;
    assert(ec_Rs.size() == ec_tracks.size());
    xarray<int> xcloud_mask = get_not_nan_index_mask(pt_cloud);
    vector<int> cloud_mask(xcloud_mask.begin(), xcloud_mask.end());     
    vector<float> errors;
    for(c= 0 ; c< ec_tracks.size(); c++){
        vector<Point3f> filtered_cloud;
        R = ec_Rs[c];
        T = ec_Ts[c];
        original_track = ec_tracks[c];
        track_mask = ec_masks[c];
        xtrack_mask = adapt(track_mask, {track_mask.size()});    
        xarray<int> intersection_mask = get_intersection_mask(cloud_mask, track_mask);     
        auto track_bool_mask = isin(xtrack_mask, intersection_mask);
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
        assert(intersection_mask.size() == xprojected_pts.size());
        assert(filter(xoriginal_pts, track_bool_mask).size() == xprojected_pts.size());
        auto xdelta = filter(xoriginal_pts, track_bool_mask) - xprojected_pts;
        assert(xdelta(0).x == filter(xoriginal_pts, track_bool_mask)(0).x - xprojected_pts(0).x);
        assert(xdelta(0).y == filter(xoriginal_pts, track_bool_mask)(0).y - xprojected_pts(0).y);
        vector<float> errors_per_frame;
        for(i = 0; i< xdelta.size(); i++){
            errors_per_frame.push_back(sqrt((xdelta(i).x)*(xdelta(i).x)+(xdelta(i).y)*(xdelta(i).y)));
        }
        errors.push_back(my_mean(errors_per_frame));    
    }
    return my_mean(errors);
}

vector<Point3f> StructureFromMotion::triangulate(Mat R1, Mat T1, Mat R2, Mat T2, vector<vector<Point2f>> track_pair){
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
    triangulatePoints(P_1, P_2, track_pair[0], track_pair[1], points4d);
    assert(points4d.cols == track_pair[0].size());
    assert(points4d.rows != 0 && points4d.cols !=0);
    points4d = points4d.t();
    Mat pts4d = points4d.reshape(4,1);
    convertPointsFromHomogeneous(pts4d, points3d);
    return points3d;
}