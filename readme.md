# Initial testing for the drone project

## vision_test.py

### Requirements:
`pip install opencv-contrib-python pupil_apriltags ultralytics`

### Description:
Uses YOLO from ultralytics to identify a person in real-time webcam video, then checks whether there is an apriltag within the bounding box of the identified person, and if so, if it corresponds to a known person. 

e.g.: 
![image](img/no_id.png)
Since the apriltag is outside of the bounding box identifying me, it is not labelled as Stella.

![image](img/person_identified.png)
Now that it is inside my bounding box, it gets labelled as Stella. 

On my machine, this runs reasonably quickly. It does not yet work properly for multiple people and I have not tested the program in varying lighting conditions.


## audio_test.py
### Requirements:
SpeechRecognition needs CUDA version 12. Probably there are better package management solutions than what I ended up doing but here's what worked.
`pip install pyaudio SpeechRecognition faster-whisper schedule gputil `
`pip install cuda-toolkit[all]==12.8`

### Description:
This transcribes the user's speech in real-time. Mostly copied from [this repo](https://github.com/davabase/whisper_real_time) modified to instead use the faster-whisper library, which claims to be faster and more memory-efficient than the OpenAI Whisper model. It also optionally logs gpu utilization information in a csv. Maybe more interesting though is watching nvitop while speaking (pip installation required) - significant spikes occur while speaking/transcribing. 

Note while running - first time using a model it will need to be downloaded, which can take a while. After it prints Model loaded, the user should speak and their words will be transcribed. I have not yet looked at the difference in accuracy between the different size models or the resource usage of each. 