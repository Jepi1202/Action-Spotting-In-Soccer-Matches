import sqlite3
import os
import json
import cv2
from datetime import datetime

class VideoDB:
    def __init__(self, path:str) -> None:
        """Constructor of the VideoDB object. Computes all the data necessary to store a video in the database.

        Args:
            path (str): Path to the mkv file
        """

        self.path = path
        # get the fps from a mkv file
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        # get the quality of the video
        self.quality = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # get the half of the video: = 1 if the path ends with '1_xxxx.mkv', = 2 if the path ends with '2_xxxx.mkv'.
        self.half = int(path[-10])
        self.training_stage = "Undefined"
        # get the duration of the video
        self.duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps)
        # get the last directory of the path
        last_dir = path.split("\\")[-2]
        self.date = last_dir[0:10]
        # convert text to date
        self.date = datetime.strptime(self.date, "%Y-%m-%d").date()

class SequenceDB:
    def __init__(self, path, data, path_video) -> None:
        self.path = path
        self.time_begin = data["gameTime"][-5:]
        # convert text to time of the form mm:ss
        self.time_begin = datetime.strptime(self.time_begin, "%M:%S").time()
        # convert time to seconds -1 because we took the images one second before the action.
        self.time_begin = self.time_begin.minute * 60 + self.time_begin.second - 1
        # Path of the video corresponds to "1_xxxx.mkv" or "2_xxxx.mkv" in the path of the sequence
        if data["gameTime"][0] == "1":
            self.path_video = os.path.join(path_video, "1_224p.mkv")
        else:
            self.path_video = os.path.join(path_video, "2_224p.mkv")
        self.label = data["label"]
        self.team = data["team"]
        # the number of images corresponds to the number of jpg files in the folder path
        self.number_of_images = len(os.listdir(path))

class ImageDB:
    def __init__(self, path, path_sequence, number) -> None:
        self.path = path
        self.path_sequence = path_sequence
        self.number = number

class DBsaver:
    """
    Class to save data in the database. The database is composed of 3 tables:
    - Video: to save the video information
    - Sequence: to save the sequence information labelled with the soccer actions and record the sound of the sequence
    - Image: to save the image information of the sequence
    """
    def __init__(self, db):
        self.db = db
    
    def createDB(path: str, name: str) -> sqlite3.Connection:
        """Create a database in the path with the name if it does not exist.

        Args:
            path (str): Path to the database
            name (str): Name of the database
        Returns:
            sqlite3.Connection: Connection to the database
        """
        db = sqlite3.connect(os.path.join(path, name))
        return db

    
    def createTables(db: sqlite3.Connection):
        """Create the 3 tables of the database.
        First table: Video(Path (key), fps, quality, half, training stage, duration, date)
        Second table: Sequence(Path (key), video path (foreign key), start time, label, team, number of images)
        Third table: Image(Path (key), sequence path (foreign key), Number)

        The duration of the video is in seconds and the start time of the sequence is in seconds.

        Args:
            db (sqlite3.Connection): Connection to the database
        """
        cursor = db.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS VIDEO(Path_video TEXT PRIMARY KEY, fps INTEGER, quality INTEGER, half INTEGER, training_stage TEXT, duration INTEGER, date TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS Sequence(Path_sequence TEXT PRIMARY KEY, VideoPath TEXT, StartTime INTEGER, Label TEXT, Team TEXT, NumberOfImages INTEGER, FOREIGN KEY(VideoPath) REFERENCES VIDEO(Path_video))")
        cursor.execute("CREATE TABLE IF NOT EXISTS IMAGE(Path TEXT PRIMARY KEY, SequencePath TEXT, Number INTEGER, FOREIGN KEY(SequencePath) REFERENCES SEQUENCE(Path_sequence))")
        db.commit()

    def insertVideo(self, path: str, quality: int, half: int, training_stage: str, duration: int, date: str, fps: int = 25):
        """Insert a video in the database. The queries pays attention to SQL injection.

        Args:
            path (str): Path to the video
            fps (int): Frames per second of the video
            quality (int): Quality of the video
            half (int): Half of the video
            training_stage (str): Training stage of the video
            duration (int): Duration of the video in seconds
            date (str): Date of the video
        """
        cursor = self.db.cursor()
        cursor.execute("INSERT INTO VIDEO VALUES(?, ?, ?, ?, ?, ?, ?)", (path, fps, quality, half, training_stage, duration, date))
        self.db.commit()
    
    def insertSequence(self, path: str, video_path: str, start_time: int, label: str, team: str, number_of_images: int):
        """Insert a sequence in the database.

        Args:
            path (str): Path to the sequence
            video_path (str): Path to the video
            start_time (int): Start time of the sequence in seconds
            label (str): Label of the sequence
            team (str): Team of the sequence
            number_of_images (int): Number of images of the sequence
        """
        cursor = self.db.cursor()
        cursor.execute("INSERT INTO SEQUENCE VALUES(?, ?, ?, ?, ?, ?)", (path, video_path, start_time, label, team, number_of_images))
        self.db.commit()
    
    def insertImage(self, path: str, sequence_path: str, number: int):
        """Insert an image in the database.

        Args:
            path (str): Path to the image
            sequence_path (str): Path to the sequence
            number (int): Number of the image in the sequence
        """
        cursor = self.db.cursor()
        cursor.execute("INSERT INTO IMAGE VALUES(?, ?, ?)", (path, sequence_path, number))
        self.db.commit()
    
    def getMkvFiles(path: str) -> list:
        """Find all mkv files in the path given or in subdirectories recursively.

        Args:
            path (str): Path where to look for mkv files.

        Returns:
            list: Path of all mkv files found starting from the given path.
        """
        
        mkvFiles = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".mkv"):
                    mkvFiles.append(os.path.join(root, file))
        return mkvFiles

    def generateVideoDB(path: str) -> VideoDB:
        """Create a VideoDB object from the path to the mkv file.

        Args:
            path (str): Path to the mkv file

        Returns:
            VideoDB: VideoDB object
        """
        return VideoDB(path)

    def insertVideoDB(self, video: VideoDB, training_stage: str = "Undefined"):
        """Insert a VideoDB object in the database.

        Args:
            video (VideoDB): VideoDB object
        """
        self.insertVideo(video.path, video.quality, video.half, training_stage, video.duration, video.date, video.fps)

    def getVideoDirectories(path: str) -> list:
        """Get all directories containing a mkv file in the path given or in subdirectories recursively.

        Args:
            path (str): Path where to look for directories.

        Returns:
            list: Path of all directories found starting from the given path.
        """
        # Attention, il faut check les doublons
        directories = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".mkv"):
                    directories.append(root)
                    break
        return directories
    
    def generateSequenceDB(path: str) -> list:
        """Create a list of SequenceDB object from the path to the directory containing the sequences.

        Args:
            path (str): Path to the directory containing the sequences.

        Returns:
            list[SequenceDB]: List of SequenceDB objects
        """
        jsonFile = os.path.join(path, "Labels-v2.json")
        # if file does not exist, return an empty list
        if not os.path.exists(jsonFile):
            return [], []
        # get the list of annotations from the json file located in a 'annotation' key of the dictionary in the json file
        with open(jsonFile, "r") as f:
            actions = json.load(f)["annotations"]
        path = os.path.join(path, "sequences")
        sequences = []
        images = []
        label_count = {}
        for action in actions:
            label = action["label"]
            if action["visibility"] == "visible":
                path_seq = os.path.join(path, label)
                if label in label_count.keys():
                    label_count[label] += 1
                else:
                    label_count[label] = 1
                
                path_seq = os.path.join(path_seq, str(label_count[label]))
                if os.path.exists(os.path.join(path_seq,"1.jpg")):
                    seq = SequenceDB(path_seq, action, path)
                    sequences.append(seq)
                    # get jpg files in the sequence directory:
                    count = 1
                    path_image = os.path.join(path_seq, str(count) + ".jpg")
                    while os.path.exists(path_image):
                        image = ImageDB(path_image, seq.path, count)
                        images.append(image)
                        count += 1
                        path_image = os.path.join(path_seq, str(count) + ".jpg")
        return (sequences, images)

    def insertSequenceDB(self, sequence: SequenceDB):
        """Insert a SequenceDB object in the database.

        Args:
            sequence (SequenceDB): SequenceDB object
        """
        self.insertSequence(sequence.path, sequence.path_video, sequence.time_begin, sequence.label, sequence.team, sequence.number_of_images)
    
    def insertImageDB(self, image: ImageDB):
        """Insert an ImageDB object in the database.

        Args:
            image (ImageDB): ImageDB object
        """
        self.insertImage(image.path, image.path_sequence, image.number)
    
    def getNbSequences(self) -> int:
        """Get the number of sequences in the database.

        Returns:
            int: Number of sequences in the database.
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM SEQUENCE")
        return cursor.fetchone()[0]

    def checkSequence(self, seq: SequenceDB) -> bool:
        """Check if the sequence is in the database.

        Args:
            seq (SequenceDB): SequenceDB object

        Returns:
            bool: True if the sequence is in the database, False otherwise.
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM SEQUENCE WHERE Path_sequence = ?", (seq.path,))
        return cursor.fetchone() is not None

    def checkImage(self, image: ImageDB) -> bool:
        """Check if the image is in the database.

        Args:
            image (ImageDB): ImageDB object

        Returns:
            bool: True if the image is in the database, False otherwise.
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM IMAGE WHERE Path = ?", (image.path,))
        return cursor.fetchone() is not None

    def updatePathVideo(self, sequences) -> None:
        
        cursor = self.db.cursor()
        params = []
        for seq in sequences:
            params.append((seq.path_video, seq.path))
        cursor.executemany("UPDATE SEQUENCE SET VideoPath = ? WHERE Path_sequence = ?", params)
        self.db.commit()


if __name__=="__main__":
    import torch
    import torchvision
    from torchvision import transforms

    # Define data transformations for data augmentation and normalization
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Define the dataset and dataloader
    data_list = [('path/to/image1.jpg', 'class1'), ('path/to/image2.jpg', 'class2'), ...]
    dataset = CustomDataset(data_list=data_list, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
