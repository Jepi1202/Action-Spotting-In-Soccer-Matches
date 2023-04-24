import os
import json
import copy
from ActionSpotting import evaluate
from TimelineInd import Timeline


class Evaluator:

    def create_json(video_path: str, timeline: Timeline, type_pred: str = "Validation"):
        actions = timeline.actions

        # Create file path
        my_path = "/scratch/users/mdelabrassinne/predictionsFolder"
        
        match_name = ""
        
        league_names = ["england_epl","europe_uefa-champions-league", "france_ligue-1","germany_bundesliga","italy_serie-a"]
        for league in league_names:
            if league in video_path:
                match_name = league
                break       
        
        season_year = ["2014-2015","2015-2016","2016-2017","2017-2018","2018-2019","2019-2020"]
        for season in season_year:
            if season in video_path:
                my_path = os.path.join(my_path,season)
                match_name = os.path.join(match_name,season)
                break
            
        match_string = copy.deepcopy(video_path[(len(my_path)+1):-11])
        match_string = match_string.replace("_"," ")
        
        my_path = os.path.join(my_path,match_string)
        match_name = os.path.join(match_name, match_string)
            
        json_dict = {}
        
        json_dict["UrlLocal"] = match_name
        json_dict["predictions"] = []

        half = video_path[-10]

        for action in actions:
            if action.getTypeAction() != "NoClass":
                name_action = action.getTypeAction()
                if action.getTypeAction() == "card":
                    name_action = "Yellow card"
                start, end = action.getTimeAction()
                time = round((start + end) / 2) # time in ms
                min = int(time/60000)
                sec = int((time/1000)%60)
                json_dict["predictions"].append({})
                gt = str(half) + " - " + str(min) + ":" + str(sec)
                json_dict["predictions"][-1]["gameTime"] = gt
                json_dict["predictions"][-1]["label"] = name_action
                json_dict["predictions"][-1]["position"] = time
                json_dict["predictions"][-1]["half"] = half
                json_dict["predictions"][-1]["confidence"] = .5
    
        json_object = json.dumps(json_dict, indent=4)

        directory = os.path.dirname(video_path)
        directory = os.path.join(directory, f"predictions_{type_pred}")
        # check if the directory exists
        if not os.path.exists(directory):
            os.mkdir(directory)
        # file is of the form '...1_224p.mkv' or '...2_224p.mkv'
        # get the half time '1' or '2'
        half = video_path[-10]
        json_file_name = os.path.join(directory, f"eval_half_{half}.json")
        
        # Writing
        with open(json_file_name, "w") as outfile:
            outfile.write(json_object)
        
        return json_file_name
    
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
    
    def write_submission(pred_files: list, sol_files: list, directory: str) -> list:
        """
        Copy all files of pred_files and sol_files in the 2 folders of the directory.

        Args:
            pred_files (list): Strings containing the paths to the json files containing the predictions
            sol_files (list): Strings containing the paths to the json files containing the ground truth
            directory (str): Path to the directory where the 2 folders will be created
        
        Returns:
            list: List of the 2 directories containing the json files
        """

        # Create the 2 directories
        directory_pred = os.path.join(directory, "predictions")
        directory_sol = os.path.join(directory, "solutions")
        if not os.path.exists(directory_pred):
            os.mkdir(directory_pred)
        if not os.path.exists(directory_sol):
            os.mkdir(directory_sol)

        # Copy the files
        i = 0
        for pred_file, sol_file in zip(pred_files, sol_files):
            pred_file_name = os.path.basename(pred_file)
            sol_file_name = os.path.basename(sol_file)
            pred_file_name = pred_file_name[:-5] + f"_{i}.json"
            sol_file_name = sol_file_name[:-5] + f"_{i}.json"
            os.system(f"cp {pred_file} {os.path.join(directory_pred, pred_file_name)}")
            os.system(f"cp {sol_file} {os.path.join(directory_sol, sol_file_name)}")
            i += 1
        
        return [directory_pred, directory_sol]
    
    def get_mAP(pred_files, sol_files, actions = ['Corner', 'Goal', 'Penalty', 'Kick-off', 'Yellow card']) -> dict:
        """
        Get the mAP of the predictions.
        Output can have the following keys:
        - mAP_classes: dict containing the mAP for each class
        - mAP: average mAP of the classes

        Args:
            pred_files (_type_): _description_
            sol_files (_type_): _description_

        Returns:
            dict: _description_
        """
        classes = ["Penalty", "Kick-off", "Goal", "Substitution", "Offside", "Shots on target", "Shots off target", "Clearance", "Ball out of play", "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick", "Corner", "Yellow card", "Red card", "Yellow->red card"]

        res = evaluate(pred_files, sol_files, version=2)

        output = {}
        output["mAP_classes"] = {}
        average = 0.
        for i, maP in enumerate(res["a_mAP_per_class_visible"]):
            if classes[i] in actions:
                output["mAP_classes"][classes[i]] = maP
                average += maP
        
        output["mAP"] = average / len(actions)
        return output