# paf_module
Peak Alpha Frequency module

This module computes sensor-space Peak Alpha Frequency (PAF) in MEG/EEG data. Alpha band oscillations - approximately 7-13Hz - are a dominant frequency in the human brain and are a great physiological marker of interindividual differences with respect to cognitive function.



Data import is done via a .csv with columns for subject ID and additionally age, diagnosis (can be commented out if desired). Though the  code for data import is simple enough to adjust to user needs.

If MEG data is to be used, it is strongly suggested to Maxwell-Filter the data beforehand (SSS, tSSS). 

