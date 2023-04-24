from TimelineInd import Timeline
import sqlite3 as sql
import cv2
import numpy as np
from tqdm import tqdm
import os
import json
from math import ceil
import torch

class Predictor:
    def __init__(self, size_window: int, model, db_path: str, tresholds: np.ndarray = None, min_count: int = None, fps: int = 25, hPixel: int = 224, wPixel: int = 224, actions: list = ["Corner", "Goal", "Card", "Penalty", "Kick-off", "NoClass"]) -> None:
        """
        Create a new predictor

        Args:
            size_window (int): The size of the window (number of frames)
            model (Model): The model to use for the prediction
            db_path (str): The path to the database
            tresholds (np.ndarray): The tresholds to use for the probabilities
            min_count (int): The minimum number of frames to consider an action in the window
            fps (int): The fps of the video
            hPixel (int): The height of the video
            wPixel (int): The width of the video
            actions (list): The list of actions
        """
        self.size = size_window
        self.model = model
        self.db_path = db_path
        # copy the tresholds in an array of size (self.size, tresholds.shape[0])
        if tresholds is None:
            tresholds = np.array([0, 0])
        self.tresholds = np.tile(tresholds, (self.size, 1))
        self.min_count = min_count
        self.n_classes = tresholds.shape[0]
        self.fps = fps
        self.hPixel = hPixel
        self.wPixel = wPixel
        self.timeline = Timeline(size_window, fps=self.fps)
        self.save = 0
        self.actions = actions
    
    def predict_matches(self, type_pred: str = "Validation", fps_pred: int = 2):
        """
        Predict all the frames of the matches in the database

        Args:
            type_pred (str, optional): Type to test. Defaults to "Validation".
            fps_pred (int, optional): Fps to use for the prediction. Defaults to 2.
        """
        c = sql.connect(self.db_path).cursor()
        c.execute("SELECT Path_video from VIDEO WHERE training_stage = ?", (type_pred,))
        matches = c.fetchall()
        for match in matches:
            print(match[0])
            path_to_save = None
            if not self.is_predicted(match[0], type_pred):
                self.save_predictions(match[0], [], type_pred, erase=True)
                self.predict_match_model(match[0], path_to_save, fps_pred)
        c.close()

    def predict_match_model(self, match_path: str, save_path: str, fps_pred: int = 2):
        """
        Predict all the frames of a match. Saves the predictions in a json file.

        Args:
            match_path (str): Path of the mkv file
            save_path (str): Path to save the predictions
            fps_pred (int, optional): Fps to use for the prediction. Defaults to 2.
        """
        cap = cv2.VideoCapture(match_path)
        preds = []
        # get the number of frames
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = ceil(self.fps / fps_pred)
        time = 0
        m = torch.tensor([0.485, 0.456, 0.406])
        s = torch.tensor([0.229, 0.224, 0.225])
        with torch.no_grad():
            for i in range(0, n_frames, self.fps):
                # frame 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                success, im = cap.read()
                if not success:
                    break
                im = [cv2.resize(im, (self.hPixel, self.wPixel))]
                im = np.array(im).reshape(1, 3, self.hPixel, self.wPixel)
                im = torch.from_numpy(im).float()
                im = im.div(255)
                im [0, 0, :] = (im[0, 0, :] - m[0]) / s[0]
                im [0, 1, :] = (im[0, 1, :] - m[1]) / s[1]
                im [0, 2, :] = (im[0, 2, :] - m[2]) / s[2]
                out = self.model(im)
                out = out.tolist()[0]
                preds.append((time, out))

                # frame 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, i + step)
                success, im = cap.read()
                if not success:
                    break

                im = [cv2.resize(im, (self.hPixel, self.wPixel))]
                im = np.array(im).reshape(1, 3, self.hPixel, self.wPixel)
                im = torch.from_numpy(im).float()
                im = im.div(255)
                im [0, 0, :] = (im[0, 0, :] - m[0]) / s[0]
                im [0, 1, :] = (im[0, 1, :] - m[1]) / s[1]
                im [0, 2, :] = (im[0, 2, :] - m[2]) / s[2]
                out = self.model(im)
                out = out.tolist()[0]
                time2 = time + ((1000 * step) // self.fps)
                preds.append((time2, out))
                time += 1000 # 1000 because we want the time in ms

                self.save += 2
                if self.save == 100:
                    print("Save:", i)
                    self.save_predictions(match_path, preds)
                    self.save = 0
                    preds = []

            self.save_predictions(match_path, preds, end=True)
    
    def is_predicted(self, match_path: int, type_pred: str = "Validation"):
        """Checks if a whole match has already been predicted

        Args:
            match_path (int): path to the mkv file
            type_pred (str, optional): Type of training stage. Defaults to "Validation".
        """

        # get directory of the match using os
        directory = os.path.dirname(match_path)
        directory = os.path.join(directory, f"predictions_{type_pred}")
        # check if the directory exists
        if not os.path.exists(directory):
            os.mkdir(directory)
        # file is of the form '...1_224p.mkv' or '...2_224p.mkv'
        # get the half time '1' or '2'
        half = match_path[-10]
        json_path = os.path.join(directory, f"half_{half}.json")

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                pred_file = json.load(f)
                return pred_file["Ended"]
        return False

    def save_predictions(self, match_path: int, pred: list, type_pred: str = "Validation", erase: bool = False, end: bool = False):
        """
        Save the predictions in a json file

        Args:
            match_path (int): path to the mkv file
            pred (list): list of predictions
            type_pred (str, optional): Type of training stage. Defaults to "Validation".
        """
        # get directory of the match using os
        directory = os.path.dirname(match_path)
        directory = os.path.join(directory, f"predictions_{type_pred}")
        # check if the directory exists
        if not os.path.exists(directory):
            os.mkdir(directory)
        # file is of the form '...1_224p.mkv' or '...2_224p.mkv'
        # get the half time '1' or '2'
        half = match_path[-10]
        json_path = os.path.join(directory, f"half_{half}.json")

        if erase:
            # remove json_path if exists
            if os.path.exists(json_path):
                os.remove(json_path)
            return

        # check if the file exists
        if os.path.exists(json_path):
            
            # load the file
            with open(json_path, "r") as f:
                pred_file = json.load(f)

            # add the new predictions
            pred_file["Time"] += [p[0] for p in pred]
            pred_file["Prediction"] += [p[1] for p in pred]
            pred_file["Ended"] = end
        else:
            pred = np.array(pred)
            pred_file = {"Match": match_path, "Time": list(pred[:, 0]), "Prediction": list(pred[:, 1]), "Ended": end}

        # create the json file
        with open(json_path, "w") as f:
            json.dump(pred_file, f)

    def load_predictions(self, type_pred: str = "Validation") -> list:
        """
        Get all the predictions from the json files if they exist (as well as the solution)

        Args:
            type_pred (str, optional): Type of training stage. Defaults to "Validation".

        Returns:
            list: list of dictionaries with the match path, the time, the predictions of each frame.
        """
        c = sql.connect(self.db_path).cursor()
        paths = c.execute("SELECT Path_video from VIDEO WHERE training_stage = ?", (type_pred,)).fetchall()
        c.close()
        
        return [self.get_predictions(path[0], type_pred) for path in paths]

    def get_predictions(self, match_path: str, type_pred: str = "Validation") -> dict:
        """
        Returns the predictions of a match if they have been created in the json file.

        Args:
            match_path (str): Path of the match
            type_pred (str, optional): Type of training stage. Defaults to "Validation".

        Returns:
            dict: Match with the information of the predictions
        """
        # get directory of the match using os
        directory = os.path.dirname(match_path)
        directory = os.path.join(directory, f"predictions_{type_pred}")
        # check if the directory exists
        if not os.path.exists(directory):
            return None
        # file is of the form '...1_224p.mkv' or '...2_224p.mkv'
        # get the half time '1' or '2'
        half = match_path[-10]
        json_path = os.path.join(directory, f"half_{half}.json")
        # check if the file exists
        if os.path.exists(json_path):
            pred_file = None
            with open(json_path, "r") as f:
                pred_file = json.load(f)
                if pred_file["Ended"]:
                    pred_file["json_proba"] = json_path
                    return pred_file
        return None
    
    def predict_match(self, predictions: dict) -> Timeline:
        """
        Predict the timeline of the actions by separating the match in windows disjoints window.

        Args:
            predictions (dict): Dictionary with the predictions of the match

        Returns:
            Timeline: The timeline of the match
        """
        # get the predictions and the time
        self.timeline = Timeline(self.size, fps=self.fps)
        preds = np.array(predictions["Prediction"])
        times = np.array(predictions["Time"])
        # get the number of windows
        n_windows = len(preds) // self.size
        # iterate over the windows
        for i in range(n_windows):
            # get the window
            window = preds[i * self.size : (i + 1) * self.size]
            # get the time
            time_window = times[i * self.size : (i + 1) * self.size]
            # get the prediction
            pred_window = self.predict_window(np.copy(window))
            # add the prediction to the timeline
            self.timeline.updateTimeline(time_window[0], pred_window)
        return self.timeline

    def predict_window(self, window: np.ndarray) -> str:
        """
        Predict the action of a window

        Args:
            window (np.ndarray): Array of vectors of probabilities

        Returns:
            str: Action of the window
        """
        window[window < self.tresholds] = 0
        window[:, -1] += 0.0001 # add a small value to the 'NoClass' action to make sure it is chosen if all probas are equal to 0.
        types_actions = np.argmax(window, axis=1)
        # get the most common action except the action 'NoClass' which has the value in types_actions : self.n_classes - 1
        nb_frames_per_action = np.bincount(types_actions[:-1])
        if nb_frames_per_action.size == 1:
            return self.actions[0]
        action = np.argmax(nb_frames_per_action)
        
        if nb_frames_per_action[action] >= self.min_count:
            return self.actions[action]
        else:
            return "NoClass"