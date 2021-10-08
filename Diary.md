# Diary of progress on Kilonova Light Curve project

## 07/10/21

The main task at the moment is to identify and codify light curve generation processes from previous papers that have done similair things in order to make training data for the Normalising flow

The first source was the DU17 model (https://iopscience.iop.org/article/10.1088/1361-6382/aa6bb0/pdf) as it seemed the simplest to turn into code.



### 13:31

I have managed to generate light curves from the equations provided with varying degrees of success. Need to experiment more with various parameters. Also managed to push the code to Github.
After lunch I will look at generating many lightcurves for just the r-band and see how that matches with expected results.

Issues:

- t_c: What is this? How important is it? Where should it roughly be?
- Graphs match the general shape but aren't perfect despite running the same code. May just be due to using too low/high mass but it is most likely an issue hidden in the functions
- theta_ej: What is a reasonable assumption? Or shouls this be given as a parameter.
- epsilon_th: Can we treat this as static? 

### 15:29

r-band dataframe created. Need to finish for today but it should be easy to try and make more data frames for the other bands.


## 08/10/2021

Half an hour before Lectures, going to make some more data.

Data frames created