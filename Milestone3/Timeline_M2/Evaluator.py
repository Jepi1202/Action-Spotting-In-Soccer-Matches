import numpy as np
import os
import json
import copy
from ActionSpotting import evaluate

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

    def evaluate_map(self, predictions: list, sol_files: list, halfs: list, dir_results: str) -> dict:
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

        assert len(predictions) == len(sol_files) == len(halfs), "The number of predictions and the number of ground truths must be the same."

        pred_files = []
        for i in range(len(predictions)):

            # write the results in a json file

            dir_match = os.path.dirname(sol_files[i])            

            predictions[i] = self.compute_nms(predictions[i])
            json_file_name = self.write_results(predictions[i], int(halfs[i]), dir_results, dir_match)
            pred_files.append(json_file_name)
        
        # Compute the Average-MaP
        return evaluate(pred_files, sol_files, version=2)

    

    def write_results(self, predictions: np.ndarray, half: int, dir_results: str, dir_match: str) -> str:
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
        json_dict["predictions"] = add_action(self, [], predictions, half)
    
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
    
    def get_json_sol(video_path: str) -> str:
        """
        Get the json file containing the ground truth of the predicitons

        Args:
            video_path (str): Path to the mkv file

        Returns:
            str: Path to the json file containing the ground truth
        """
        directory = os.path.dirname(video_path)
        json_file = os.path.join(directory, "Labels-v2.json")
        return json_file

