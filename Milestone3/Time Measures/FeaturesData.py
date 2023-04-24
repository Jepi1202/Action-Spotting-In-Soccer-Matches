import numpy as np


def getPathFeatures(path_game):
    """
        Get the features of a game

        Args:
        :param path_game: path of a football game
        :param features: list of features (input to the model)
    """
    # The section 7 in the path of the features is composed of ' ' instead of '_'
    parties = path_game.split('/')
    parties[7] = parties[7].replace("_", " ")

    path_game = f"{parties[0]}/{parties[1]}/{parties[2]}/{parties[3]}/{parties[4]}/{parties[5]}/{parties[6]}/{parties[7]}/{parties[8]}"

    # There are 2 kind of videos: 224p.mkv and 720p.mkv
    if '224p.mkv' in path_game:
        baidu_path = path_game.replace('224p.mkv', 'baidu_soccer_embeddings.npy')
    elif '720p.mkv' in path_game:
        baidu_path = path_game.replace('720p.mkv', 'baidu_soccer_embeddings.npy')
    else:
        raise ValueError('Video is not in the right format')
    
    #print(baidu_path)
    return baidu_path


def getFeatures(features_path):
    """
        Get the features of a game

        Args:
        :param features_path: path of the features
    """
    return np.load(features_path)


def getFeaturesWindow(features, window_size, index):
    """
        Get the features of a window

        Args:
        :param windowSize: size of the window (in seconds)
        :param index: index of the window
    """
    
    if (index + 1) * window_size > len(features): 
        return features[index * window_size: len(features),:]
    return features[index * window_size: (index+1) * window_size - 1, :]
