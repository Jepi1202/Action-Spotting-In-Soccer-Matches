from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pyplot as plt
from FeaturesData import getPathFeatures, getFeatures, getFeaturesWindow
import os
import cv2
from moviepy.editor import *
import imageio
import av

def measureTimeWindow(path_game, window_size):
    """
        Measure the time to set the input to the model and to get the output

        Args:
        :param path_game: path of a football game
        :param window_size: size of the window (in seconds)
        :param features: list of features (input to the model)
    """
    # Get the Baidu features of the game
    features_path = getPathFeatures(path_game)
    features = getFeatures(features_path)

    measures = []
    max = len(features) // window_size

    for i in range(0, max):
        start = time.time()

        # Get the features of the window
        features_window = getFeaturesWindow(features, window_size, i)

        # set to the model the features of the window

        end = time.time()
        measures.append(end - start)

    return measures.mean()

def measureTimeWindows(game_path, window_sizes):
    """
        Measure the time to set the input to the model and to get the output for different window sizes

        Args:
        :param game_path: path of a football game
        :param window_sizes: list of window sizes (in seconds)
        :param features: list of features (input to the model)
    """
    measures = []
    for window_size in window_sizes:
        measures.append(measureTimeWindow(game_path, window_size))

    return measures

def displayMeasureTime(window_sizes, measures):
    """
        Display the time to set the input to the model and to get the output

        Args:
        :param game_path: path of a football game
        :param window_sizes: list of window sizes (in seconds)
        :param features: list of features (input to the model) 
    """
    fig, ax = plt.subplots()
    ax.plot(window_sizes, measures)
    ax.set_xlabel('Window size')
    ax.set_ylabel('Time (s)')
    plt.grid()
    plt.show()
    
    
def videoToFrames_1(video_path, output_dir):
    """
    From a mkv file, extract the frames and save them in a folder with cv2.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Failed to open the video.")
        exit()
    
    i = 0
    while cap.isOpened():
        is_read, frame = cap.read()
        #if i >= 1000:
        #    break
        if is_read:
            frame_filename = os.path.join(output_dir, "frame_{}.png".format(i))
            cv2.imwrite(frame_filename, frame)
            i += 1
        else:
            break
    cap.release()
    
    
def videoToFramesTIME_1(video_path):
    output_dir = 'Test_Frames'
    
    start = timer()
    videoToFrames_1(video_path, output_dir)
    end = timer()
    
    if os.path.exists(output_dir):
        # Remove the folder
        import shutil
        shutil.rmtree(output_dir)
        print("Folder {} removed.".format(output_dir))

    return (timedelta(seconds=end-start))
               
def videoToFrames_2(video_path, output_dir):
    """
        From a mkv file, extract the frames and save them in a folder with moviepy.editor.
    """

    video = VideoFileClip(video_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop to extract frames
    for i, frame in enumerate(video.iter_frames()):
        #if i >= 1000:
        #    break
        frame_filename = os.path.join(output_dir, "frame_{}.png".format(i))
        with open(frame_filename, "wb") as f:
            f.write(frame)
        
    print("Frames extraction ended.")
    
    
def videoToFramesTIME_2(video_path):
    output_dir = 'Test_Frames'
    
    start = timer()
    videoToFrames_2(video_path, output_dir)
    end = timer()
    
    if os.path.exists(output_dir):
        # Remove the folder
        import shutil
        shutil.rmtree(output_dir)
        print("Folder {} removed.".format(output_dir))

    return (timedelta(seconds=end-start))

def videoToFrames_3(video_path, output_dir):
    """
        From a mkv file, extract the frames and save them in a folder with imageio.
    """

    video = imageio.get_reader(video_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop to extract frames
    for i, frame in enumerate(video):
        if i >= 500:
            break
        frame_filename = os.path.join(output_dir, "frame_{}.png".format(i))
        imageio.imwrite(frame_filename, frame)
        
    print("Frames extraction ended.")
    
    
def videoToFramesTIME_3(video_path):
    output_dir = 'Test_Frames'
    
    start = timer()
    videoToFrames_3(video_path, output_dir)
    end = timer()
    
    if os.path.exists(output_dir):
        # Remove the folder
        import shutil
        shutil.rmtree(output_dir)
        print("Folder {} removed.".format(output_dir))

    return (timedelta(seconds=end-start))

def videoToFrames_4(video_path, output_dir):
    """
        From a mkv file, extract the frames and save them in a folder with av.
    """
    container = av.open(video_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop to extract frames
    for frame in container.decode(video =0):
        if frame.index >= 1500:
            break
        image = frame.to_image()
        image.save(output_dir + '/frame_{}.png'.format(frame.index))
        
    container.close()
    print("Frames extraction ended.")
    
    
def videoToFramesTIME_4(video_path):
    output_dir = 'Test_Frames'
    
    start = timer()
    videoToFrames_4(video_path, output_dir)
    end = timer()
    
    if os.path.exists(output_dir):
        # Remove the folder
        import shutil
        shutil.rmtree(output_dir)
        print("Folder {} removed.".format(output_dir))

    return (timedelta(seconds=end-start))