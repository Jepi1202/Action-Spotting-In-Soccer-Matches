class Timeline:

    def __init__(self) -> None:
        self.actions = []
        self.actions.append(Timeline.Action(0, "None"))
        self.lastTimeStep = 0
        pass

    def process_actions(self):
        """Process the actions in the timeline. If an action appears several times in 10 seconds, they will be merged. An action should be deleted if it last less than 1 second after the merge. If several different types of action overlap, the overlapping part will be deleted for the one starting last until there is no overlapping left.
        """
        for i in range(len(self.actions)):
            if not self.actions[i].processed:
                next = i
                break
        
        # Merge actions
        while next < len(self.actions):
            if not self.actions[next].processed:
                self.actions[next].processed = True
                i = next+1
                while i < len(self.actions):
                    if self.actions[i].type_action == self.actions[next].type_action:
                        if self.actions[i].start - self.actions[next].end <= 10000: # 10 seconds in milliseconds
                            self.actions[next].end = self.actions[i].end
                            self.actions.pop(i)
                            i-=1
                        else:
                            break
                    i += 1

                if self.actions[next].end - self.actions[next].start < 1000: # 1 second in milliseconds
                    self.actions.pop(next)
                else:
                    next += 1
            else:
                next += 1
        
        # print all actions
        for i in range(len(self.actions)):
            print(self.actions[i].start, self.actions[i].end, self.actions[i].type_action)
        
        # Delete overlapping actions
        i = 0
        while i < len(self.actions):
            if not self.actions[i].processed:
                raise Exception("Action not processed")
            j = i+1
            while j < len(self.actions):
                if self.actions[i].end > self.actions[j].start:
                    self.actions[i].end = self.actions[j].start
                j += 1
            i += 1
        
        # Add action NoClass when there is no action
        i = 0
        while i < len(self.actions):
            if i == 0:
                if self.actions[i].start != 0:
                    self.actions.insert(0, Timeline.Action(0, "NoClass"))
                    self.actions[0].end = self.actions[1].start
                    i += 1
            else:
                if self.actions[i].start - self.actions[i-1].end > 0:
                    self.actions.insert(i, Timeline.Action(self.actions[i-1].end, "NoClass"))
                    self.actions[i].end = self.actions[i+1].start
                    i += 1
            i += 1
        if self.actions[-1].end != self.lastTimeStep:
            self.actions.append(Timeline.Action(self.actions[-1].end, "NoClass"))
            self.actions[-1].end = self.lastTimeStep
                

    def updateTimeline(self, time_action: int, type_action: str):
        """
        Update the timeline with the action and the time of the action. The time must always be greater than the one from the last call of the method.

        Args:
            time_action (int): The time of the action
            type_action (str): The class of the action
        """
        prev = self.actions[-1]
        if prev.getTypeAction() != type_action:
            self.actions.append(Timeline.Action(time_action, type_action))
        else:
            prev.update(time_action)
        self.lastTimeStep = time_action
    
    class Action:
        """
        Class that represents an action in the timeline
        """

        def __init__(self, time_action_begin: int, type_action: str) -> None:
            """
            Constructor of the class

            Args:
                time_action_begin (int): The start of the action
                time_action_end (int): The end of the action
                type_action (str): The class of the action
            """
            self.start = time_action_begin
            self.end = time_action_begin
            self.type_action = type_action
            self.processed = False
        
        def update(self, end_time) -> None:
            self.end = end_time

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

if __name__=="__main__":
    timeline = Timeline()
    timeline.updateTimeline(100, "NoClass")
    timeline.updateTimeline(200, "NoClass")
    timeline.updateTimeline(300, "NoClass")
    timeline.updateTimeline(400, "NoClass")
    timeline.updateTimeline(500, "NoClass")
    timeline.updateTimeline(600, "NoClass")
    timeline.updateTimeline(700, "NoClass")
    timeline.updateTimeline(800, "NoClass")
    timeline.updateTimeline(900, "NoClass")
    timeline.updateTimeline(1000, "NoClass")
    timeline.updateTimeline(1100, "NoClass")
    timeline.updateTimeline(1200, "NoClass")
    timeline.updateTimeline(1300, "NoClass")
    timeline.updateTimeline(1400, "NoClass")
    timeline.updateTimeline(1500, "NoClass")
    timeline.updateTimeline(1600, "NoClass")
    timeline.updateTimeline(1700, "NoClass")
    timeline.updateTimeline(1800, "Goal")
    timeline.updateTimeline(1900, "Corner")
    timeline.updateTimeline(2000, "NoClass")
    timeline.updateTimeline(2100, "NoClass")
    timeline.updateTimeline(2200, "Corner")
    timeline.updateTimeline(2300, "Corner")
    timeline.updateTimeline(2400, "NoClass")
    timeline.updateTimeline(2500, "Foul")
    timeline.updateTimeline(2600, "Goal")
    timeline.updateTimeline(2700, "NoClass")
    timeline.updateTimeline(2800, "NoClass")
    timeline.updateTimeline(2900, "NoClass")
    timeline.updateTimeline(3000, "Foul")
    timeline.updateTimeline(3100, "Corner")
    timeline.updateTimeline(3200, "Goal")
    timeline.updateTimeline(3300, "NoClass")
    timeline.updateTimeline(3400, "NoClass")
    timeline.updateTimeline(3500, "NoClass")
    timeline.updateTimeline(3600, "NoClass")
    timeline.updateTimeline(3700, "Foul")
    timeline.updateTimeline(3800, "NoClass")
    timeline.updateTimeline(3900, "NoClass")
    timeline.updateTimeline(4000, "NoClass")
    timeline.updateTimeline(4100, "NoClass")
    timeline.updateTimeline(4200, "NoClass")
    timeline.updateTimeline(4300, "NoClass")
    timeline.updateTimeline(4400, "NoClass")
    timeline.updateTimeline(4500, "NoClass")
    timeline.updateTimeline(4600, "NoClass")
    timeline.process_actions()
    
    print("\nFinal\n")
    for action in timeline.actions:
        
        print(action.getTimeAction(), action.getTypeAction())