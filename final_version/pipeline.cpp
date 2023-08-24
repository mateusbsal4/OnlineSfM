#include "definitions.h"
#include "pipeline.h"

int main(int argc, char ** argv){
    StructureFromMotion sfm;
    if(argc==1){
        VideoCapture cap(0);
        return sfm.runSfM(cap);
    }
    else{
        string path_to_vid = "/home/mateus/IC/OnlineSfM/Dataset/" + (string)argv[1] + ".MOV"; 
        VideoCapture cap(path_to_vid);
        return sfm.runSfM(cap);
    }
}

StructureFromMotion::StructureFromMotion(){

}

int StructureFromMotion::runSfM(VideoCapture cap){
    viz::Viz3d window("SfM Visualization");
    window.setWindowSize(Size(2000,2000));
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
            assert(init_Rs.size() == 5);
            if(!cloud_3d.empty()){
                error = calculate_init_error(error_calc_tracks, error_calc_masks, cloud_3d);       
            }
            if(error > error_threshold){
                dropped_tracks += 1;
            }
            else{
                is_init = 0;
            }
            continue;
        }
        vector<vector<Point2f>> remaining_tracks(tracks.end() - error_calculation_frames-1, tracks.end());
        vector<vector<int>> remaining_masks(masks.end() - error_calculation_frames-1, masks.end());                       
        vector<int> remaining_frame_numbers(frame_numbers.end()-error_calculation_frames-1, frame_numbers.end()); 
        tie(global_Rs, global_Ts, cloud_3d) = reconstruct(remaining_tracks, remaining_masks, global_Rs, global_Ts, cloud_3d); 
        if(global_Rs.size() % ba_window == 0){         //frame selected for BA
            run_ba(global_Rs, global_Ts, cloud_3d, tracks, masks);      //reference to global objects is passed, so they are modified inside run_ba function
            adjust_path = 1;
        }
        viz::WCloud cloud_widget(cloud_3d, viz::Color::white());
        cloud_widget.setRenderingProperty(viz::POINT_SIZE, 3.0);
        window.showWidget("point_cloud", cloud_widget);
        if(global_Rs.size() > 0){  
            if(adjust_path){        //this corrects the trajectory after each ba run   
                for(size_t path_ctr = path.size() + 1 - ba_window; path_ctr < path.size(); path_ctr++){
                    path[path_ctr] = Affine3d(global_Rs[path_ctr], global_Ts[path_ctr]);
                }
                adjust_path = 0;
            }
            for(size_t path_counter = path.size(); path_counter<global_Rs.size(); path_counter ++){
                path.push_back(Affine3d(global_Rs[path_counter], global_Ts[path_counter]));
            }
            window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::PATH, 0.1, viz::Color::blue()));
            window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K_viz, 0.1, viz::Color::red()));                   
        }
        window.spinOnce(1, true);          
    }
    while(1){window.spinOnce(1, true);}     //keeps the window open after pipeline has ended
    return 0;
}