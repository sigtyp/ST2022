# PRECOR-Transformer: a system to PREdic COgnate Reflexes

The folder contains:
* `run.py`, the file that can be used to train the transformer (`python run.py` or `python run.py --surprise`, if the training set is the surprise languages)
* the `surprise` folder with the predicted values for each "surprise language"
* the `training` folder with the predicted values for the languages of the original training set

The model can be trained tuning three main hyperparameters:

`python run.py --embed_dimensions 150 -latent_dimensions 800 --heads 5 --batch_size 400`

The values of the hyperparameters shown are the default ones if they are not changed. 

## Acknowledgements

The model has been created by Giuseppe G. A. Celano within the DFG project 408121292.
