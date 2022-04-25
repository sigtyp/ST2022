# Mockingbird System N2

Only the results for 10% data sparsity are provided.

Cognate Neighbors model, where a single model is constructed for each language
group
([implementation](https://github.com/google-research/google-research/tree/master/cognate_inpaint_neighbors)).

Caveat: Please note, different models are trained for different number of steps
with no meaningful stopping criteria. After some reasonably large number of
steps the training is stopped and the inference is performed using the latest
checkpoint. The checkpoint is documented under the inference section in the
respective `README.md` files for each language group.
