class Timeline:

    class Action:
        """
        Class that represents an action in the timeline
        """

        def __init__(self, start: int, end: int, type_action: str) -> None:
            """ 
            Create a new action

            Args:
                start (int): Start time of the action
                end (int): End time of the action
                type_action (str): Class of the action
            """
            self.start = start
            self.end = end
            self.type_action = type_action

        def getTimeAction(self) -> tuple:
            """
            Get the time of the action

            Returns:
                int: The time of the action
            """
            return self.start, self.end

        def getTypeAction(self) -> str:
            """
            Get the class of the action

            Returns:
                str: The class of the action
            """
            return self.type_action

    def __init__(self, size_window: int, fps: int = 25) -> None:
        """
        Constructor of the class

        Args:
            size_window (int): The size of the window in number of frames
            fps (int, optional): The fps of the video. Defaults to 25.
        """
        self.size = size_window
        self.fps = fps
        self.actions = []
        

    def updateTimeline(self, start: int, type_action: str):
        """
        Update the timeline with the action and the time of the action. The time must always be greater than the one from the last call of the method.

        Args:
            start (int): Start time of the action
            type_action (str): The class of the action
        """
        if start == 0:
            self.actions.append(Timeline.Action(start, start + self.size * 1000 / self.fps, type_action))
            return
        # The window must be the same and no overlap can happen
        if start != self.actions[-1].end:
            print("\n\n")
            for action in self.actions:
                print(action.getTimeAction(), action.getTypeAction())
            print(start, type_action)
            raise Exception("The time of the action must be the same than the one from the last call of the method")
        
        # Probably the action was seperated in 2. We already stored the previous one so no need to store it again.
        if type_action == self.actions[-1].type_action:
            self.actions.append(Timeline.Action(start, start + self.size * 1000 / self.fps, "NoClass"))
        else:
            self.actions.append(Timeline.Action(start, start + self.size * 1000 / self.fps, type_action))

    