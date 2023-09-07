# SfM Pipeline - Structure from Motion

This is the repository for an incremental Structure from Motion (SfM) pipeline capable of processing video frames in real-time. The pipeline recovers the 3D structure of a scene and the camera's trajectory as the video progresses. This README will guide you through setting up the project and running it.

## Dependencies

Before running the SfM pipeline, you need to install the following dependencies:

- [Eigen3](https://eigen.tuxfamily.org/dox/GettingStarted.html) (3.3 or higher)
- [OpenCV](https://opencv.org/) (4.0 or higher)
- [xtensor](https://github.com/xtensor-stack/xtensor) (Included as a submodule)
- [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas) (Included as a submodule)
- [Ceres Solver](http://ceres-solver.org/) (1.14 or higher)

## Building the Project

1. Clone this repository to your local machine:

    ```shell
    git clone https://github.com/your-username/final_version.git
    ```

2. Navigate to the project directory:

    ```shell
    cd final_version
    ```

3. Create a build directory:

    ```shell
    mkdir build
    cd build
    ```

4. Generate the project files with CMake:

    ```shell
    cmake ..
    ```

5. Build the project:

    ```shell
    make
    ```

## Running the SfM Pipeline

Once you've built the project successfully, you can run the SfM pipeline. Make sure you have a video file to process.

1. Execute the pipeline with the following command:

    ```shell
    ./final_version /path/to/your/video.mp4
    ```

   Replace `/path/to/your/video.mp4` with the path to your video file.

2. The pipeline will start processing the video frames, and you will see the 3D structure and camera trajectory visualization as the video progresses.

