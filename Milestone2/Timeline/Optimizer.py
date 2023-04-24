import numpy as np
import os
from Predictor import Predictor
from Evaluator import Evaluator


class Optimizer:
    def __init__(self, window: list, count: list, thresholds: dict, db_path: str = "/scratch/users/mdelabrassinne/Database/SoccerDB.db", fps: int = 2, type_pred: str = "Validation", actions_name: list = ['Corner', 'Goal', 'Penalty', 'Kick-off','card', 'NoClass']) -> None:
        """
        Class constructor for Optimising the mAP of the matches.

        Args:
            window (list): List containing the window sizes for the optimizer.
            count (list): List containing the number minimm of frames in a window so that the window is considered as an event.
            thresholds (dict): List of elements size "Classes + 1"
            db_path (str, optional): Path to the database. Defaults to "/scratch/users/mdelabrassinne/Database/SoccerDB.db".
            fps (int, optional): Number of frames per second. Defaults to 2.
            type_pred (str, optional): Type of prediction. Defaults to "Validation".
        """
        self.window = window
        self.count = count
        self.thresholds = thresholds
        self.db_path = db_path
        self.fps = fps
        self.type_pred = type_pred
        self.actions_name = actions_name
    

    def set_new_params(self, window: list = None, count: list = None, thresholds: dict = None) -> None:
        """
        Set new parameters for the optimizer.

        Args:
            window (list, optional): Params of window. Defaults to None.
            count (list, optional): Params of count. Defaults to None.
            thresholds (dict, optional): Params of tresholds. Defaults to None.
        """
        if window is not None:
            self.window = window
        if count is not None:
            self.count = count
        if thresholds is not None:
            self.thresholds = thresholds
        
    def create_timelines(self, window: int, count: int, thresholds: np.ndarray):
        """
        Create the timelines for the matches.

        Args:
            window (int): Size of the window.
            count (int): Number of frames in a window so that the window is considered as an event.
            thresholds (np.ndarray): List of thresholds per class

        """
        predictor = Predictor(window, None, self.db_path, tresholds=thresholds, min_count = count, fps=self.fps, actions = self.actions_name)
        preds = predictor.load_predictions()
        real_preds = []
        for pred in preds:
            if pred:
                real_preds.append(pred)
        preds = real_preds
        evals = []
        for pred in preds:
            timeline = predictor.predict_match(pred)
            evals.append({"Match": pred["Match"], "Timeline": timeline})
        return evals


    def create_predictions(self, evals: list):
        pred_files = []
        sol_files = []
        for eval in evals:
            eval["json_pred"] = Evaluator.create_json(eval["Match"], eval["Timeline"], self.type_pred)
            eval["json_sol"] = Evaluator.get_json_sol(eval["Match"])
            # if sol does not exist, remove eval from evals
            if os.path.exists(eval["json_sol"]):
                # remove eval from evals
                pred_files.append(eval["json_pred"])
                sol_files.append(eval["json_sol"])
        return pred_files, sol_files

    

    def optimize_window(self, count: int, thresholds: np.ndarray) -> list:
        """
        Optimizes the window size for the mAP of the matches.

        Args:
            count (int): Number of frames in a window so that the window is considered as an event.
            thresholds (np.ndarray): List of thresholds per class

        Returns:
            list: List containing the values of the mAP per param Value of type (Window, mAP)
        """
        mAP = []
        for w in self.window:
            evals = self.create_timelines(w, count, thresholds)
            pred_files, sol_files = self.create_predictions(evals)
            res = Evaluator.evaluate(pred_files, sol_files)
            mAP.append((w, res))
        return mAP

    def optimize_count(self, window: int, thresholds: np.ndarray) -> list:
        """
        Optimizes the count size for the mAP of the matches.

        Args:
            window (int): Size of the window.
            thresholds (np.ndarray): List of thresholds per class

        Returns:
            list: List containing the values of the mAP per param Value of type (count, mAP)
        """
        mAP = []
        for c in self.count:
            evals = self.create_timelines(window, c, thresholds)
            pred_files, sol_files = self.create_predictions(evals)
            res = Evaluator.evaluate(pred_files, sol_files)
            mAP.append((c, res))
        return mAP

    def optimize_threshold(self, window: int, count: int, thresholds, threshold: str) -> list:
        """
        Optimizes the threshold value for the mAP of the matches.

        Args:
            window (int): Size of the window.
            count (int): Number of frames in a window so that the window is considered as an event.
            thresholds (np.ndarray): List of thresholds per class
            threshold (str): Name of the threshold to optimize.

        Returns:
            list: List containing the values of the mAP per param Value of type (threshold, mAP)
        """

        # check if threshold exists
        if threshold not in self.thresholds.keys():
            raise Exception("Threshold does not exist")
        
        mAP = []
        for t in self.thresholds[threshold]:
            # get position of threshold
            pos = self.actions_name.index(threshold)
            thresholds[pos] = t
            evals = self.create_timelines(window, count, thresholds)
            pred_files, sol_files = self.create_predictions(evals)
            res = Evaluator.evaluate(pred_files, sol_files)
            mAP.append((t, res))
        return mAP
            


