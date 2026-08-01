import random

class QA_database:
    
    def __init__(self):
        self.description_database = [
        "Describe the anomaly events observed in the video.",
        "Could you describe the anomaly events observed in the video?",
        "Could you specify the anomaly events present in the video?",
        "Give a description of the detected anomaly events in this video.",
        "Could you give a description of the anomaly events in the video?",
        "Provide a summary of the anomaly events in the video.",
        "Could you provide a summary of the anomaly events in this video?",
        "What details can you provide about the anomaly in the video?",
        "How would you detail the anomaly events found in the video?",
        "How would you describe the particular anomaly events in the video?",
    ]
        
        self.analysis_database = [
            "Why do you judge this event to be anomalous?",
            "Can you provide the reasons for considering it anomalous?",
            "Can you give the basis for your judgment of this event as an anomaly?",
            "What led you to classify this event as an anomaly?",
            "Could you provide the reasons for considering this event as abnormal?",
            "What evidence do you have to support your judgment of this event as an anomaly?",
            "Can you analyze the factors contributing to this anomalous event?",
            "Could you share your analysis of the anomalous event?",
            "What patterns did you observe that contributed to your conclusion about this event being an anomaly?",
            "How do the characteristics of this event support its classification as an anomaly?",
        ]
        self.severity_database = [
            "On a scale of 0–4, how severe is the anomaly in this video?",
            "Rate the anomaly’s severity from 0 (none) to 4 (critical).",
            "Assign a severity score (0–4) to the anomaly shown.",
            "What severity level (0–4) would you give this anomaly?",
            "Please evaluate the anomaly’s severity on a 0–4 scale.",
            "Provide a 0–4 severity rating for the anomaly.",
            "How would you score the anomaly’s severity (0–4)?",
            "Choose a severity level for the anomaly: 0–4.",
            "Give the anomaly a severity label on the 0–4 scale.",
            "Estimate the anomaly’s severity using a 0–4 rating.",
        ]
        self.category_database_new = [
            "Does this video contain any anomalies? If yes, what type(s) of anomaly are present?",
            "Identify whether there are anomalies in this video. If there are, assign each anomaly to an appropriate category.",
            "Determine if the video shows any anomalies. If it does, indicate the category of each anomaly.",
            "Check the video for anomalies. If any are present, describe them and specify their categories.",
            "Examine the video and determine whether it contains anomalies. If so, list the anomalies and their categories.",
            "Analyze the video for potential anomalies. If anomalies are found, provide the category for each.",
            "Assess whether the video includes any anomalous events. If it does, classify each event into a category.",
            "Inspect the video for anomalies. If any are detected, state their categories.",
            "Determine if the video segment contains anomalous behavior or events. If so, categorize each one.",
            "Review the video to identify anomalies. If anomalies are present, report them along with their categories.",
        ]
        self.category_database = [
            "What types of anomalies are shown in the video clip?",
            "Can you classify the anomaly into a specific category?",
            "Could you identify the category of the anomaly in the video?",
            "Please specify the category of the anomaly observed in the video.",
            "How would you categorize the anomaly present in this video?",
            "What type of anomaly is depicted in the video?",
            "Can you determine the category of the anomaly shown in this video?",
            "Could you provide the classification of the anomaly in the video?",
            "What is the appropriate category for the anomaly observed in this video?",
            "How would you define the category of the anomaly present in this video?",
        ]
        self.category_context = """
        #   You are a classification system for identifying anomalous driving behaviors.  
            There are **32 predefined anomaly categories** listed below.  
            When asked a question, **reply only with the corresponding category number** (no text, punctuation, or explanation).  

        Anomaly Categories:     1 change of lane
                                2 late turn
                                3 cutting inside turns
                                4 driving on the centerline
                                5 yielding to emergency vehicles
                                6 brief wait at an open intersection
                                7 long wait at an empty intersection
                                8 too far onto the main road while waiting
                                9 stopping at an unusual point
                                10 slowing at an unusual point
                                11 fast driving that appears reckless
                                12 slow driving with apparent uncertainty
                                13 unusual movement pattern
                                14 brief reverse movement
                                15 unusual approach toward waiting or slow cars
                                16 traffic tie up
                                17 almost cut another traffic agent off
                                18 cut another traffic agent off clearly
                                19 almost collision
                                20 into oncoming lane while turning
                                21 illegal turn
                                22 short wrong way in roundabout then exit
                                23 wrong way driver
                                24 more than one full turn in a roundabout
                                25 broken down vehicle on street
                                26 stop mid street to let a person cross
                                27 stop at a crosswalk to let a person cross
                                28 slight departure from the roadway
                                29 on or parking on sidewalk
                                30 strong sudden braking
                                31 swerve to avoid or maneuver around a vehicle
                                32 risky behaviour that does not fit another category
"""
        self.four_category_context = """
        #   You are a classification system for identifying anomalous driving behaviors.  
            There are **predefined anomaly categories** listed below. 
            1: "speed_trajectory_irregularities":Abnormal speed choice or unstable movement patterns that raise risk (or clearly deviate from "normal" driving), even if they don't directly create a near-collision event.
            2: "direction_space_violations":Violations of intended direction of travel or legal use of space, such as entering the oncoming lane, performing illegal turns, or occupying sidewalks.
            3: "conflict_near_collision":Events where interaction with another road user becomes critical: one vehicle clearly cuts off another, a near crash occurs, or emergency maneuvers (hard braking, swerving) are taken to avoid collision.
            4: "stopped_obstruction_right_of_way":Vehicles that are stopped or nearly stopped in the roadway for special reasons—emergency vehicles, breakdowns, letting pedestrians cross—or that act as an obstruction.

        Respond only with the category name! Do not include any number, explanation, punctuation, formatting, or extra text.
"""
        self.cat_abc_sys_prompt = """
        You are a classification system for identifying anomalous driving behaviors in traffic videos.
        
        There are 5 possible categories. Each category is represented by a SINGLE special label token:
        
        - A: No anomaly. All visible road users behave normally.
        - B: Abnormal speed choice or unstable trajectory (e.g., sudden acceleration/deceleration, weaving, too fast or too slow driving) that increases risk.
        - C: Wrong-way driving, illegal or unsafe turns, lane/space violations (e.g., driving outside lanes, cutting across islands, using sidewalk/shoulder).
        - D: Near-collision or collision events where road users come into critical conflict (cut-offs, forced hard braking or swerving, crash).
        - E: Vehicles stopped or nearly stopped in the roadway or blocking right-of-way (e.g., stopped in lane, breakdown, obstruction).
        
        When you answer, you MUST:
        
        1. Look at the whole video and choose the single most appropriate category.
        2. Output **exactly one** of the 5 categories above.
        3. Output **nothing else** (no explanations, no natural language, no extra spaces or punctuation).
        """

        self.category_ABC_sys_prompt_NEW = """
        You are a video classification system that identifies anomalous driving behavior in traffic videos.

        You must assign exactly one label token from the set {A, B, C, D, E} based on all visible road users across the entire video. Choose the single best label using the definitions below. If multiple anomalies occur, output only the most severe and more relevant one.
        
        Labels (output exactly one token):

        - A: No anomaly. All visible road users behave normally.
        - B: Abnormal speed choice that increases risk (too fast or too slow relative to traffic, road, or conditions).
        - C: Direction or maneuver violation, including wrong-way driving, illegal or unsafe turns, and illegal or unsafe lane changes.
        - D: Near-collision or collision event, including cut-offs that force hard braking or evasive action, near-misses, and crashes.
        - E: Abnormal road use, including stopping in an active lane or mid-road (breakdown, yielding to emergency vehicles, letting pedestrians cross) and driving or parking in non-vehicle areas (sidewalk, shoulder, etc.).

        When you answer, you MUST:
        1. Watch the entire video before deciding.
        2. Output exactly one label token only: A or B or C or D or E.
        3. Output nothing else: no words, no punctuation, no extra whitespace, no newlines.

        Hint: If multiple anomalies are present, choose the one cause the anomaly event. e.g. if a wrong-way driver causes a near-collision, choose C over D.
        """

        self.category_ABC_sys_prompt_4cat = """
        You are a video classification system that identifies anomalous driving behavior in traffic videos.

        You must assign exactly one label token from the set {A, B, C, D} based on all visible road users across the entire video. Choose the single best label using the definitions below. If multiple anomalies occur, output only the most severe and more relevant one.
        
        Labels (output exactly one token):

        - A: No anomaly. All visible road users behave normally.
        - B: Direction or maneuver violation, including wrong-way driving, illegal or unsafe turns, and illegal or unsafe lane changes.
        - C: Near-collision or collision event, including cut-offs that force hard braking or evasive action, near-misses, and crashes.
        - D: Abnormal road use, including stopping in an active lane or mid-road (breakdown, yielding to emergency vehicles, letting pedestrians cross) and driving or parking in non-vehicle areas (sidewalk, shoulder, etc.).

        When you answer, you MUST:
        1. Watch the entire video before deciding.
        2. Output exactly one label token only: A or B or C or D.
        3. Output nothing else: no words, no punctuation, no extra whitespace, no newlines.
        """
        
        self.description_sys_prompt = """
        You are an expert driving behavior analyst. Your task is to describe anomalous driving events in clear, detailed natural language.

        For each video clip, produce a description that includes all of the following:

        1) Road and environment context  
        Describe the road type and scene layout, such as roundabout, intersection, straight multilane road, highway, residential street, or parking lot. Include any relevant environmental or traffic context when visible, such as weather, lighting, road markings, traffic signs, and traffic density.

        2) Target object grounding  
        Clearly identify the anomalous road user as “object A.” State its type (for example car, truck, bus, motorcycle, bicycle, pedestrian) and describe where it appears and how it moves relative to the camera view, for example “object A enters from the left side of the frame and moves toward the camera in the inner lane.”

        3) Anomaly event description
        Describe in detail what object A does that is unusual, unsafe, or non compliant with normal traffic rules. 

        Output your answer as one concise paragraph of two to four sentences that covers all three parts above. If multiple anomalies are visible, focus on the most severe or most clearly visible anomalous event.
        """

        self.description_analysis_database = [
        "Describe and analyze the anomaly events observed in the video.",
        "Could you describe and analyze the anomaly events observed in the video?",
        "Give a description and analysis of the detected anomaly events in this video.",
        "Could you give a description and analysis of the anomaly events in the video?",
        "Provide a summary and analysis of the anomaly events in the video.",
    ]
        

    def question_selection(self,type):
        if type == "description":
            return random.choice(self.description_database)
        elif type == "analysis":
            return random.choice(self.analysis_database)
        elif type == "severity":
            return random.choice(self.severity_database)
        elif type == "category":
            return random.choice(self.category_database)
        elif type == "category_new":
            return random.choice(self.category_database_new)
        elif type == "description_analysis":
            return random.choice(self.description_analysis_database)

    def question_type_query(self, question: str) -> str:
        """Return which question type the input belongs to."""
        if question in self.description_database:
            return "description"
        elif question in self.analysis_database:
            return "analysis"
        elif question in self.severity_database:
            return "severity"
        elif question in self.category_database:
            return "category"
        elif question in self.category_database_new:
            return "category"
        else:
            return "unknown"

    def cat_context(self):
        return self.category_context

    def four_cat_context(self):
        return "You are a classification system for identifying anomalous driving behaviors. \nThere are 5 predefined anomaly categories listed below. \n0: \"no anomaly\" \n1: \"speed_trajectory_irregularities\": Abnormal speed choice or unstable movement patterns that raise risk (or clearly deviate from “normal” driving), even if they don’t directly create a near-collision event. \n2: \"direction_space_violations\": The main problem is where the car is relative to legal lanes/roadway/sidewalk, including wrong-way, illegal turns, and using space not meant for vehicles. \n3: \"conflict_near_collision\": Events where interaction with another road user becomes critical: one vehicle clearly cuts off another, a near crash occurs, or emergency maneuvers (hard braking, swerving) are taken to avoid collision. \n4: \"stopped_obstruction_right_of_way\": Vehicles that are stopped or nearly stopped in the roadway for special reasons—emergency vehicles, breakdowns, letting pedestrians cross—or that act as an obstruction. \n\nReply only the category name.\n"
        # return self.four_category_context

    def category_to_index(self, category_name: str) -> str:
        categories = {
            "change of lane": "1",
            "late turn": "2",
            "cutting inside turns": "3",
            "driving on the centerline": "4",
            "yielding to emergency vehicles": "5",
            "brief wait at an open intersection": "6",
            "long wait at an empty intersection": "7",
            "too far onto the main road while waiting": "8",
            "stopping at an unusual point": "9",
            "slowing at an unusual point": "10",
            "fast driving that appears reckless": "11",
            "slow driving with apparent uncertainty": "12",
            "unusual movement pattern": "13",
            "brief reverse movement": "14",
            "unusual approach toward waiting or slow cars": "15",
            "traffic tie up": "16",
            "almost cut another traffic agent off": "17",
            "cut another traffic agent off clearly": "18",
            "almost collision": "19",
            "into oncoming lane while turning": "20",
            "illegal turn": "21",
            "short wrong way in roundabout then exit": "22",
            "wrong way driver": "23",
            "more than one full turn in a roundabout": "24",
            "broken down vehicle on street": "25",
            "stop mid street to let a person cross": "26",
            "stop at a crosswalk to let a person cross": "27",
            "slight departure from the roadway": "28",
            "on or parking on sidewalk": "29",
            "strong sudden braking": "30",
            "swerve to avoid or maneuver around a vehicle": "31",
            "risky behaviour that does not fit another category": "32",
        }

        # Normalize input for case and whitespace
        category_name = category_name.strip().lower()
        return categories.get(category_name, None)

    def four_cat_to_index(self, category_name: str) -> str:
        categories = {
            "no anomalies": "0",
            "no anomaly": "0",
            "speed_trajectory_irregularities": "1",
            "direction_space_violations": "2",
            "conflict_near_collision": "3",
            "stopped_obstruction_right_of_way": "4",
        }

        # Normalize input for case and whitespace
        category_name = category_name.strip().lower()

        if category_name in categories:
            return categories[category_name]
        else:
            return category_name