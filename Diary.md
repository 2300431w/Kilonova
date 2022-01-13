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

```
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
```
# 19/10/2021

## 13:00
Worked to ensure that the full flow of the model of the paper was followed. This required reworking some of the functions but the process was relatively smooth. However, when this new and "improved" code was run the graph was even further from the expected values than before! This is likely due to a mis understanding in my code or a paramater. The lead culprit is the kappa variable in the M_X function. Some quick testing showed that it could indeed result in higher and lower graph values so I am planning to do some dimensional analysis on the formula it appears in to make sure that I am using the correct value. It is initially given in 10 cm^2 g^-1 which might result in a totally different value when converted to meters, kg etc. 

## 16:00
Even when converted kappa was = 1 m^2/kg which helped very marginally but not by much. I have several functions I need to check before I can move on.


# 21/10/2021
## 14:37
The first of the issues solved was by plotting with regards to t' and not t. This was not terribly clear in the paper but it has worked to improve the closeness of the predictions to those from the paper. To better locate the issue I first plotted the Bolometric correction functions to compare to those in the paper. As we can see bellow they were exceptionally close:
![alt text](https://github.com/2300431w/Kilonova/blob/master/BC%20Curves.png)
![alt text](https://github.com/2300431w/Kilonova/blob/master/BC%20Curves%20Goal.PNG)

From this we can clearly see that any inconsistency with the Magnitude graphs is not the source of the issues. However when comparing the Bolometric luminosity predictions by the paper and my program we can see significant difference:

![alt text](https://github.com/2300431w/Kilonova/blob/master/L_bol%20vs%20time.png)
![alt text](https://github.com/2300431w/Kilonova/blob/master/L_bol%20vs%20time%20GOAL.PNG)

The shapes are approximately accurate but there is a clear discrepency on the y-axis. I will try to investigate potential problems in the luminosity function. At least I know where the issue is now and that at least one part is correct

# 02/11/2021

Last wednesday I emailed professor Tim Dietrich about the model in the paper he co-authored and he kindly provided a link to the github page. The format of how their application works wasn't terribly clear so I have spent the past week trying to do three things. The first was to try and reconcile my old code using their functions when possible. Second was to try and build a new code base based on their code and the final (and ultimatley succesful) challenge: Search the source code for usable functions to use and put together to make their code work in a convenient way for my project. The other two methods prodced inconsistent results with the other paper for reasons I am not quite sure. The final method's results were far more reliable and used very little of my own code (rather it was old code being rearanged and used in a new way by me). This looks promising and will hopefully provide a useful basis for generating training data. As of right now the function takes in the masses of the NS (m1,m2) and the compactness (c1,c2). I need to ensure that these are variables we can assume will be known quickly enough to be useful in this context or if I will have to add another stage where I calculate c1 and c2 or maybe even m1 and m2 from the GW detection (though such an  in depth analysis of the GW signal seems beyond the scale of this project but you never know).


# 23/11/2021

Rather than compactness I needed to use the llamda values instead. Fortunatley there is an EOS agnostic relation (CLove). 

Thanks to m1,m2,l1,l2 combinations from Jordan McGinn I was able to create 100k lightcurves. This was then used to train a normalised Flow AI. So far I have only trained with the g-band of the spectrum (with r,i,z yet to be done) but the proof of concept is there and in theory it shouldn't be difficult to change which spectrum is being trained. See below an example of a train/validation loss from initial g-band training:

![alt_text](https://github.com/2300431w/Kilonova/blob/master/Train_Val_Loss.png)

This suggests some overfitting. The hyper parameters used are: 

```
flow = RealNVP(
    n_inputs=901,
    n_transforms =4,
    n_conditional_inputs=4,
    n_neurons=32,
    batch_norm_between_transforms=True)
    
batch_size = 1000
val_split = 0.33
epochs = 100
```

These are the produced graphs: 

![alt_text](https://github.com/2300431w/Kilonova/blob/master/AI_gband_1.png)

With the red line representing the average values while the blue dots are reported from individual lines for values of m1,m2,l1,l2 chosen randomly from the original dataset

# Winter Break

I added noise at some point but it didn't fix the fundamental issues, I have contacted Michael and Jordan over it however over winter break I need to catch up on studies and similair things

# 11/01/2022

## 12:45
Under the advice of Michael and Jordan I have reduced the number of dimensions by "compressing" the data. Here we are taking the 10th point of every line, Jordan mentioned PCA (Principle Component Analysis) so I should look into that. Currently reducing the resolution of the curve has seemed to help but the examples drawn at the end don't seem close at all. The loss bottoms out at approximatley 83 regardless of batch size,learning rate, or epoch length so perhaps that is indicitive that this approach isn't working.

I have also normalised the curves which I was not doing before. I might try a further compression but I am concerned that anything beyond a factor of 10 will lose too much detail.

# 13/01/2022

## 10:35
I need to look into primary component analysis and build new data from it. I'm not sure what the best way to do this is so I have contacted Jordan to see how he suggests it is used with flow models. While I wait I will try to find a way to introduce learning rate scheduling and see if that improves the model.

## 15:34 
With all the data in a single file the model had loss in the regions of e16 and upwards. When I turn on BatchNorm = True however the loss dropped significantly. learning rate scheduling also seems to be very effective. Currently I am reducing the rate with gamma = 0.1 every 20%*epochs 
