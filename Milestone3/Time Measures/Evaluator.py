import numpy as np
import os
import json
import copy
from ActionSpotting import evaluate
import torch

class Evaluator:
    """
    Evaluates the Average-MaP of the predictions of the Yahoo model for SoccerNet using NMS on time series.
    """

    def __init__(self, fps: int, nms_window: list, classes: list) -> None:
        """
        Constructor of the Evaluator class.

        Args:
            fps (int): Frame per second of the predictions.
            nms_window (list): Size in seconds of the NMS window for each class
            classes (list): List of the classes.
        """
        self.fps = fps
        self.nms_window = nms_window
        self.classes = classes
        self.nb_classes = len(classes)

    def evaluate_map(self, models, features_half1: list, features_half2: list, input_size: int, labels: list, dir_results: str) -> dict:
        """
        Evaluates the Average-MaP of the predictions of the Yahoo model for SoccerNet using NMS on time series.

        Args:
            model (): Models used to make the predictions based on yahoo architecture.
            features_half1 (list): Baidu features of the first half of each video.
            features_half2 (list): Baidu features of the second half of each video.
            input_size (int): Length of the sequence used to make the predictions.
            labels (list): List of JSON files containing the labels of each video.
            dir_results (str): Directory where the results will be saved.

        Returns:
            dict: Dictionary containing the Average-MaP and the MaP for each class.
        """

        def compute_confidences(self, features):
            conf = np.zeros((features.shape[0], self.nb_classes))
            # Make the predictions
            for i in range(0, features.shape[0], input_size):
                # Make the predictions
                if i+input_size >= features.shape[0]:
                    seq = features[-input_size:]
                    seq = np.expand_dims(seq, axis=0)
                    seq = torch.Tensor(seq)
                    seq = seq.transpose(1, 2)
                    confidences = ((models[0](seq)).detach().numpy()).transpose()
                    confidences = np.squeeze(confidences, axis=2)
                    confidences = 1 / (1 + np.exp(-confidences))
                    displacements = ((models[1](seq)).detach().numpy()).transpose()
                    displacements = np.squeeze(displacements, axis=2)
                    end = features.shape[0] - i
                    conf[i:] = self.merge_nms(confidences[-end:], displacements[-end:])
                else:
                    seq = features[i:i+input_size]
                    seq = np.expand_dims(seq, axis=0)
                    seq = torch.Tensor(seq)
                    seq = seq.transpose(1, 2)
                    confidences = ((models[0](seq)).detach().numpy()).transpose()
                    confidences = np.squeeze(confidences, axis=2)
                    confidences = 1 / (1 + np.exp(-confidences))
                    displacements = ((models[1](seq)).detach().numpy()).transpose()
                    displacements = np.squeeze(displacements, axis=2)
                    conf[i:i+input_size] = self.merge_nms(confidences, displacements)
            return conf

        assert len(features_half1) == len(features_half2) == len(labels), "The number of features and labels should be the same."

        pred_files = []
        for i in range(len(features_half1)):
            feat1 = np.load(features_half1[i])
            feat2 = np.load(features_half2[i])
            ground_truths = labels[i]

            conf1 = compute_confidences(self, feat1)
            conf2 = compute_confidences(self, feat2)

            # write the results in a json file
            dir_match = os.path.dirname(labels[i])
            json_file_name = self.write_results(conf1, conf2, dir_results, dir_match)
            pred_files.append(json_file_name)
        
        # Compute the Average-MaP
        return evaluate(pred_files, labels, version=2)

    

    def write_results(self, conf1: np.ndarray, conf2: np.ndarray, dir_results: str, dir_match: str) -> str:
        """
        Writes the results in a JSON file.

        Args:
            confidences (np.ndarray): Updated confidences of each class for each time step of each video.
            ground_truths (list): List of JSON files containing the labels of each video.
            dir_results (str): Directory where the results will be saved.
            dir_match (str): Directory of the match.
        """

        def add_action(self, actions: list, confidences: np.ndarray, half: int) -> list:
            for i in range(len(confidences)):
                probas = confidences[i]
                time = i / self.fps
                min = int(time/60)
                sec = int((time)%60)
                for c, proba in enumerate(probas):
                    if proba != 0:
                        gt = str(half) + " - " + str(min).zfill(2) + ":" + str(sec).zfill(2)
                        name_action = self.classes[c]
                        actions.append({})
                        actions[-1]["label"] = name_action
                        actions[-1]["position"] = time * 1000 # time in millseconds
                        actions[-1]["half"] = half
                        actions[-1]["confidence"] = proba
                        actions[-1]["gameTime"] = gt
            
            return actions

        # Create file path  
        my_path = dir_results     
        match_name = ""
        
        league_names = ["england_epl","europe_uefa-champions-league", "france_ligue-1","germany_bundesliga","italy_serie-a"]
        for league in league_names:
            if league in dir_match:
                match_name = league
                break       
        
        season_year = ["2014-2015","2015-2016","2016-2017","2017-2018","2018-2019","2019-2020"]
        for season in season_year:
            if season in dir_match:
                my_path = os.path.join(my_path,season)
                match_name = os.path.join(match_name,season)
                break
            
        match_string = copy.deepcopy(dir_match[(len(my_path)+1):-11])
        match_string = match_string.replace("_"," ")
        
        my_path = os.path.join(my_path,match_string)
        match_name = os.path.join(match_name, match_string)
            
        json_dict = {}
        
        json_dict["UrlLocal"] = match_name
        json_dict["predictions"] = add_action(self, [], conf1, 1)
        json_dict["predictions"] = add_action(self, json_dict["predictions"], conf2, 2)
    
        json_object = json.dumps(json_dict, indent=4)

        json_file_name = os.path.join(dir_match, f"{match_name}.json")
        # match name equals to path of last folder
        match_name = match_name.split("/")[-1]
        
        # Writing
        json_file_name = os.path.join(dir_results, f"{match_name}.json")
        
        # Writing
        with open(json_file_name, "w") as outfile:
            outfile.write(json_object)
        
        return json_file_name

    def merge_nms(self, confidences: np.ndarray, displacements: np.ndarray) -> np.ndarray:
        """
        Evaluates the updated confidences using the displacements and Normalized Maximum Suppression (NMS) on time series.

        Args:
            confidences (np.ndarray): Confidence of each class for each time step of each video.
            displacements (np.ndarray): Displacement of each class for each time step of each video. The displacements 
                are in seconds.
        Returns:
            np.ndarray: Updated confidences of each class for each time step of each video.
        """

        # Check the shape of the confidences and displacements
        assert confidences.shape == displacements.shape, "The shape of the confidences and displacements should be the same."

        # To compute the Average-MaP, 3 steps are needed:
        # 1. Compute the updated confidences using the displacements
        # 2. Compute the NMS on the time series
        # 3. Compute the MaP for each class and the Average-MaP (This step required additional computations)

        # 1. Compute the updated confidences using the displacements
        updated_confidences = self.compute_update_confidence(confidences, displacements)

        # 2. Compute the NMS on the time series
        nms_confidences = self.compute_nms(updated_confidences)
        return nms_confidences
        

    def compute_update_confidence(self, confidences: np.ndarray, displacements: np.ndarray) -> np.ndarray:
        """
        Computes the updated confidences using the displacements.

        Args:
            confidences (np.ndarray): Confidence of each class for each time step of each video.
            displacements (np.ndarray): Displacement of each class for each time step of each video.

        Returns:
            np.ndarray: Updated confidences of each class for each time step of each video.
        """
        T = confidences.shape[0]
        K = confidences.shape[1]

        displacements = np.round(displacements * self.fps).astype(int)
        output = np.zeros_like(confidences)
        classes = np.arange(K)

        for t in range(T):
            disp = displacements[t, :]
            idx = t - disp
            idx = np.clip(idx, 0, T - 1)
            output[idx, classes] = np.maximum(output[idx, classes], confidences[t, classes])
        
        return output


    def compute_nms(self, confidences: np.ndarray) -> np.ndarray:
        """
        Computes the NMS on the time series for each class seperately.

        Args:
            confidences (np.ndarray): Updated confidences of each class for each time step of each video.

        Returns:
            np.ndarray: NMS confidences of each class for each time step of each video.
        """
        T = confidences.shape[0]
        K = confidences.shape[1]

        output = confidences.copy()

        for k in range(K):
            nms_window = self.nms_window[k]
            nms_window = int(nms_window * self.fps)
            preds = confidences[:, k]
            copy_preds = preds.copy()

            # Apply NMS for a window of size nms_window seconds
            max_idx = np.argmax(copy_preds)
            max_val = copy_preds[max_idx]

            while max_val > 0:
                start = max(0, max_idx - nms_window)
                end = min(T, max_idx + nms_window + 1)

                preds[start:end] = 0
                preds[max_idx] = max_val

                copy_preds[start:end] = 0
                max_idx = np.argmax(copy_preds)
                max_val = copy_preds[max_idx]
            
            output[:, k] = preds
        
        return output

