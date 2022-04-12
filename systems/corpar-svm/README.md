# Demonstration of how a System can be Added to the Shared Task

This little demo system shows how the system by [List et al. (forthcoming)](https://arxiv.org/abs/2204.04619) can be used to analyze the data in the shared task. 

To install all packages needed to run the code, make a fresh virtual environment in Python and run the following in your terminal (we use Python 3.9):
```
$ pip install -r requirements.txt
```

To then run the code for the training data, just type:

```
$ python run.py
```

To run the code for the surprise data, type:

```
$ python run.py --surprise
```

The results will be written to the files in the folders `training` and `surprise`, respectively. 

