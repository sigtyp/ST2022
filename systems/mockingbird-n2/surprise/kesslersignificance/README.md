# `kesslersignificance`

## Data generation

```shell
python neighbors/data/create_neighborhood.py --output_dir /tmp/tmp --max_rand_len 10 --language_group kesslersignificance --lang all --pairwise_algo lingpy --logtostderr --task_data_dir ~/projects/ST2022/data-surprise  --num_duplicates 100
```

## Training

```shell
python neighbors/model/trainer.py  --run_locally=cpu --model=feature_neighborhood_model_config.TransformerWithNeighborsTiny --input_symbols /tmp/tmp/kesslersignificance.syms --output_symbols /tmp/tmp/kesslersignificance.syms --feature_neighborhood_train_path tfrecord:/tmp/tmp/kesslersignificance_train.tfrecords --feature_neighborhood_dev_path tfrecord:/tmp/tmp/kesslersignificance_test.tfrecords --feature_neighborhood_test_path tfrecord:/tmp/tmp/kesslersignificance.tfrecords --learning_rate=0.001 --max_neighbors=4 --max_pronunciation_len=11 --max_spelling_len=9 --logdir /tmp/logdir/
```

## Inference

```shell
python neighbors/model/decoder.py --feature_neighborhood_test_path=tfrecord:/tmp/tmp/kesslersignificance_test.tfrecords --input_symbols /tmp/tmp/kesslersignificance.syms --output_symbols /tmp/tmp/kesslersignificance.syms --ckpt /tmp/logdir/train --model=TransformerWithNeighborsTiny --decode_dir /tmp/tmp/decode_dir/ --split_output_on_space --max_neighbors=4 --max_pronunciation_len=11 --max_spelling_len=9 --batch_size 1
```

Latest model from `/tmp/logdir/train/ckpt-00025947`.
