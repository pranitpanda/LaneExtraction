# LaneExtraction
Code from 2019 summer research project at UCSD. Uses OpenCV to extract lanes and navigate autonomously. Integrated into DonkeyCar framework

Object avoider is the final product. It contains code that extracts the lanes from the images passed in by the Raspberry Pi camera. It then calculates the correct steering inputs needed to follow the lanes and passes those to the motor controller. If an object appears in the road, the car swerves to avoid it. Once the object is gone, the lane following algorithm will resume, allowing the car to return to its original path after it has passed the object. For testing convinence and due to budget constraints, I defined the object to be blue and use a mask to identify blue objects. However, with more cameras and sensors, any color object could be avoided using similar strategies. In that way the color does not affect the solvability of the problem, which is why I chose to just solve the problem of blue objects as a proof of concept.

Helper_functions contains just the opencv functions for isolating the lanes and calculating the cars position relative to it. Also contains function for derving needed steering inputs based on the car position. Basically does everything except actually drive the car.

Countour and ContourUp both contain the full driving functionality of the car which implements the lane extraction techinques detailed in helper_function file

RedLight.py contains regular lane-extraction code and driving functionality and stops at red lights
