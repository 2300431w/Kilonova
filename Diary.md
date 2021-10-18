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

Data frames created. Im not sure if this data is 100% reliable. It doesnt seem to match the expected graph. 

## 12/10/2021

Data created in every frequency band availle using the model above. Though the underlying function might not be perfect it will hopefully allow me to at least get a framework for working with the Normalised flow machine. 

There was little issue creating the data but I decided to use the pandas dataframe model. Hopefully this will just future proof things but might not actually be the perfect method

##  14/10/2021

After a bit of work I have the Glasflow model taking the data input and attempting to learn from it. As expected the loss decreases quite rapidly before reaching ~0.58 
First attempt used a very high number of iterations and didn't show much improvment in loss (though it appeared very low, not sure if this was an error however). Eventually after a few thousand iterations the loss would oscillate around the initial value, suggesting it probably wasn't learning much anymore.

In the next attempts I lowered the number of iterations and deviated from the example by putting the flow.eval() funciton to run every 30 iterations along with the print progress update. This resulted in a strange phenomena of the loss starting as before ~0.58 and then jumping incredibly high (~6,000) before beginning the exponential drop as expected back down to 0.57. This might be worth following up on to ensure I understand what flow.eval() is doing or why the initial loss is so low.

Loss per iteration on attempt 2:

'''
iteration 0      t_loss: 0.576261579990387
iteration 30     t_loss: 62857.578125
iteration 60     t_loss: 338.20013427734375
iteration 90     t_loss: 232.23345947265625
iteration 120    t_loss: 59.77543640136719
iteration 150    t_loss: 21.80772590637207
iteration 180    t_loss: 11.824857711791992
iteration 210    t_loss: 6.902825355529785
iteration 240    t_loss: 4.394950866699219
iteration 270    t_loss: 3.0358095169067383
iteration 300    t_loss: 2.2417237758636475
iteration 330    t_loss: 1.7466562986373901
iteration 360    t_loss: 1.421630620956421
iteration 390    t_loss: 1.1995223760604858
iteration 420    t_loss: 1.0429623126983643
iteration 450    t_loss: 0.9299145936965942
iteration 480    t_loss: 0.8467209339141846
iteration 510    t_loss: 0.7845571637153625
iteration 540    t_loss: 0.7375448942184448
iteration 570    t_loss: 0.7016433477401733
iteration 600    t_loss: 0.6740162968635559
iteration 630    t_loss: 0.6526268124580383
iteration 660    t_loss: 0.6359855532646179
iteration 690    t_loss: 0.6229921579360962
iteration 720    t_loss: 0.6128208041191101
iteration 750    t_loss: 0.6048431396484375
iteration 780    t_loss: 0.5985807776451111
iteration 810    t_loss: 0.5936625003814697
iteration 840    t_loss: 0.5898008942604065
iteration 870    t_loss: 0.5867711305618286
iteration 900    t_loss: 0.5843971967697144
iteration 930    t_loss: 0.5825404524803162
iteration 960    t_loss: 0.5810912847518921
iteration 990    t_loss: 0.5799630284309387
'''
