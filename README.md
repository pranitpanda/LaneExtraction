# LaneExtraction
Code from 2019 summer research project at UCSD. Uses OpenCV to extract lanes and navigate autonomously. Integrated into DonkeyCar framework

Helper_functions contains just the opencv functions for isolating the lanes and calculating the cars position relative to it. Also contains function for derving needed steering inputs based on the car position. Basically does everything except actually drive the car.

Countour and ContourUp both contain the full driving functionality of the car which implements the lane extraction techinques detailed in helper_function file

RedLight.py contains regular lane-extraction code and driving functionality and stops at red lights
