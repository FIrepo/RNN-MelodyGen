# RNN-MelodyGen

Description:
------------
This is a melody generator using recurrent neural network (RNN).

Requirements:
-------------
Python 2 or 3.  
Theano

How to use:
-----------
*     Gather midi files as training data, put them into ./midi  
*     Run convert_music.py. Converted files are in ./converted  
*     Train the network and yield its Op.1. Commands:  
      from run import *  
      run(mode='new',iteration=15)  
*     The network is now stored in param_note.txt. To compose new pieces,  
      run(mode='file')  
      -Note: New midi will overwrite the old one.  
    
*     You can modify parameters to make it perform well, such as size of network.  
      Parameters of network->run.py (size,iteration)  
      Parameters of generating midi->songbird.py (speed,length)  
     
