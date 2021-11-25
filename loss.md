# Validation vs Training Loss

Training Data length: 100k

Hyper Parameters and setup: 
```
# Training Data length: 100k

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

The Loss vs validation curves are:
![alt text](https://github.com/2300431w/Kilonova/blob/master/Train_Val_Loss_2.png)

Where I think There are signs of overfitting as the validation loss begins to climb up or at least not continue to fall as the train loss does


comparing the Flow generated data to the model predictions (time [days] on the x-axis and magnitude on the y):
![alt text](https://github.com/2300431w/Kilonova/blob/master/g-band%20flow%20vs%20model.png)

where the indices [0] and [1] represent random choices from the shuffled dataset
