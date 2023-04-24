import sqlite3
import DBsave as dbs

class DBmanager(dbs.DBsaver):
    """
    Class to manage data in the database. The database is composed of 3 tables:
    - Video: to save the video information
    - Sequence: to save the sequence information labelled with the soccer actions and record the sound of the sequence
    - Image: to save the image information of the sequence
    """
    def __init__(self, pathDB: str, nameDB: str):
        self.db = sqlite3.connect(pathDB + nameDB)
        self.cursor = self.db.cursor()
    
    def getVideo(self, path: str) -> tuple:
        """Get the video information from the database.

        Args:
            path (str): Path to the video
        Returns:
            tuple: Video information
        """
        self.cursor.execute("SELECT * FROM VIDEO WHERE Path_video = ?", (path,))
        return self.cursor.fetchone()