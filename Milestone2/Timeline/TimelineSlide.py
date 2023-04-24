class Timeline:

    def __init__(self, size_window: int, size_step: int, fps: int = 25) -> None:
        """
        Constructor of the class
        Args:
            size_window (int): The size of the window in number of frames
            size_step (int): The size of the step of the window in number of frames
            fps (int, optional): The fps of the video. Defaults to 25.
        """
        self.size = size_window
        self.step = size_step
        self.fps = fps
        self.actions = []
        self.types_actions = ["Kick-off"]

        self.lastTimeStep = 0

    def updateTimeline(self, start: int, action: str) -> None:
        """
        Update the timeline with the new action and the time it started.
        The time must always be greater than the last time added.

        Args:
            start (int): Time the action started.
            action (int): Action that was performed.
        """
        
        if len(self.actions) > 0:
            prev_action = self.actions[-1]
            if prev_action.getStart() > start:
                raise ValueError(
                    "The time must be greater than the last time added.")

        self.actions.append(Action(start, start + self.size * 1000 / self.fps, action))
        self.types_actions.append(action)
        self.lastTimeStep = start + self.size

    def mergeActions(self) -> None:
        """
        Merge the actions that are close in time.

        - If an action appears several times in 10 seconds, they will be merged.
        - An action should be deleted if it lasts less than 1 second.
        - If several types of action overlap, the overlapping part will be
        deleted for the one starting last until there is no overlapping left.
        (end1 = start2 if start2 before end1)

        """
        # Maximum time between two actions to be merged
        max_time = 10000

        # the intial number of actions.
        nb_actions = len(self.actions)
        action_index = 0

        while action_index < nb_actions:
            action = self.actions[action_index]  # get the current action
            # Last action in the list
            if action == self.actions[-1]:
                break

            # get the index of the current action in the list
            ind = action_index

            # Mergde actions that are close in time
            while (ind < nb_actions - 1) and (action.getEnd() + max_time >= self.actions[ind + 1].getStart()):
                next_action = self.actions[ind + 1]
                if action.getTypeAction() == next_action.getTypeAction():
                    action.setEnd(next_action.getEnd())
                    self.actions.remove(next_action)
                    self.types_actions.remove(next_action.getTypeAction())
                    nb_actions -= 1
                    
                else:
                    ind += 1
            
            # delete the actions that are too short (less than 1 sec)
            if action.getEnd() - action.getStart() < 1000:
                self.actions.remove(action)
                self.types_actions.remove(action.getTypeAction())
                nb_actions -= 1
                action_index -= 1

            action_index += 1

        # Delete overlapping actions
        for action in self.actions:
            if action != self.actions[-1]:
                next_action = self.actions[self.actions.index(action) + 1]

                if action.getEnd() > next_action.getStart():
                    action.setEnd(next_action.getStart())

        # Add action NoClass when there is no action
        self.addNoClassAction()

        #print("------------------")
        #for action in self.actions:
        #    print(action.getStart(), action.getEnd(), action.getTypeAction())


    def addNoClassAction(self) -> None:
        """
        Add action NoClass when there is no action.
        """
        nb_actions = len(self.actions)

        # If all the actions are too short --> deleted OR any action detected => NoClass on all the video
        if nb_actions <= 0:
            self.actions.append(0,Action(0, self.lastTimeStep, "NoClass"))
            return
        
        action_index = 0
        while action_index < nb_actions - 1 :
            if action_index == 0 and self.actions[action_index].getStart() != 0:
                self.actions.insert(
                    0, Action(0, self.actions[action_index].getStart(), "NoClass"))
                self.types_actions.insert(0, "NoClass")
                nb_actions += 1
                action_index += 1
            else:

                if self.actions[action_index].getEnd() != self.actions[action_index + 1].getStart():
                    self.actions.insert(action_index + 1, Action(self.actions[action_index].getEnd(
                    ), self.actions[action_index + 1].getStart(), "NoClass"))
                    self.types_actions.insert(action_index + 1, "NoClass")
                    nb_actions += 1
                    action_index += 1

            action_index += 1

        if self.actions[-1].getEnd() != self.lastTimeStep:
            self.actions.append(
                Action(self.actions[-1].getEnd(), self.lastTimeStep, "NoClass"))
            self.types_actions.append("NoClass")

    
class Action:

    def __init__(self, start: int, end: int, action: str) -> None:
        """
        Create a new action.
        Args:
            `start` time the action started.
            `end` time the action ended.
            `action` type of the action that was performed.
        """
        self.start = start
        self.end = end
        self.type_action = action
        self.processed = False

    def getStart(self) -> int:
        """
        Return the start of the action.

        Returns:
        -------
            `int`  the start time of the action took.
        """
        return self.start

    def getEnd(self) -> int:
        """
        Return the end time of the action.

        Returns:
        -------
            `int`  time the action ended.
        """
        return self.end

    def setEnd(self, end: int) -> None:
        """
        Set the end time of the action.

        Parameters:
        ----------
            `end` time the action ended.
        """
        self.end = end

    def setStart(self, start: int) -> None:
        """
        Set the start time of the action.

        Parameters:
        ----------
            `start` time the action started.
        """
        self.start = start

    def getTypeAction(self) -> str:
        """
        Return the class of the action.

        Returns:
        -------
            `str`  the class of the action.
        """
        return self.type_action

    def getTimeAction(self) -> tuple:
        """
        Get the time of the action
        Returns:
            int: The time of the action
        """
        return self.start, self.end

# test
if __name__ == "__main__":
    print("Initialisation du test de timeline")

    # Create a timeline
    timeline = Timeline(10, 10)
    timeline.updateTimeline(4000, "NoClass")
    timeline.updateTimeline(4600, "NoClass")
    timeline.updateTimeline(5000, "NoClass")
    timeline.updateTimeline(5400, "NoClass")
    timeline.updateTimeline(5800, "NoClass")
    timeline.updateTimeline(6200, "NoClass")

    
    timeline.mergeActions()
