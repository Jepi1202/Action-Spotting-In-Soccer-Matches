import cv2

class Match():
    def __init__(self, mkvFile: str, comments_frequency: int = 2, fps: int = 25, quality: list = [224, 224]) -> None:
        """Constructor for Match class representing a soccer match...

        Args:
            mkvFile (str): mkv file path.
            comments_frequency (int, optional): Frequency of comments in seconds Defaults to 2.
            fps (int, optional): Frames per second. Defaults to 25.
            quality (list, optional): Quality of the images. Defaults to [224, 224].
        """
        self.mkvFile = mkvFile
        self.comments_frequency = comments_frequency
        self.sequence = 0
        self.time_begin_min = 0
        self.time_begin_sec = 0
        self.time_end_min = 0
        self.time_end_sec = 0
        self.fps = fps
        self.match_length = -1
        self.quality = quality

    def getImagesFromNextSequence(self) -> list:
        """Get images from next sequence in the movie soccer.

        Returns:
            list: List of images.
        """
        self.updateBounds()
        self.sequence += 1
        images = self.getImagesFromBounds()
        return images

    def getImagesFromBounds(self) -> list:
        """Get the images from the bounds of the match representing a sequence.

        Returns:
            list: List of images in the sequence.
        """
        cap = cv2.VideoCapture(self.mkvFile)
        cap.set(cv2.CAP_PROP_POS_MSEC, (self.time_begin_min * 60 + self.time_begin_sec) * 1000)
        images = []
        nb_images = self.comments_frequency * self.fps
        for i in range(nb_images):
            ret, frame = cap.read()
            if not ret:
                break
            images.append(frame)
        return images
    
    def updateBounds(self) -> None:
        """update bounds of the sequence of the match.
        """
        self.time_begin_min = self.time_end_min
        self.time_begin_sec = self.time_end_sec
        self.time_end_min = self.time_begin_min + self.comments_frequency // 60
        self.time_end_sec = self.time_begin_sec + self.comments_frequency % 60
    

    def getMatchLength(self) -> int:
        """Get the length of the match in seconds.

        Returns:
            int: Length of the match in seconds.
        """
        if self.match_length == -1:
            cap = cv2.VideoCapture(self.mkvFile)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self.fps
            self.match_length = length
        return self.match_length