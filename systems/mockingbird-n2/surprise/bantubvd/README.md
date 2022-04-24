# `bantubvd`

## Data generation

```shell
python neighbors/data/create_neighbors.py --output_dir /tmp/tmp --max_rand_len 10 --language_group bantubvd --lang all --pairwise_algo lingpy --random_target_algo markov --logtostderr --task_data_dir ~/projects/ST2022/data-surprise  --num_duplicates 50
```

## Training

```shell
python neighbors/model/trainer.py  --run_locally=cpu --model=feature_neighbors_model_config.TransformerWithNeighborsTiny --input_symbols /tmp/tmp/bantubvd.syms --output_symbols /tmp/tmp/bantubvd.syms --feature_neighbors_train_path tfrecord:/tmp/tmp/bantubvd_train.tfrecords --feature_neighbors_dev_path tfrecord:/tmp/tmp/bantubvd_test.tfrecords --feature_neighbors_test_path tfrecord:/tmp/tmp/bantubvd_test.tfrecords --learning_rate=0.001 --max_neighbors=9 --max_pronunciation_len=11 --max_spelling_len=3 --logdir /tmp/logdir/
```

## Inference

```shell
python neighbors/model/decoder.py --feature_neighbors_test_path=tfrecord:/tmp/tmp/bantubvd_test.tfrecords --input_symbols /tmp/tmp/bantubvd.syms --output_symbols /tmp/tmp/bantubvd.syms --ckpt /tmp/logdir/train --model=TransformerWithNeighborsTiny --decode_dir /tmp/tmp/decode_dir/ --split_output_on_space --max_neighbors=9 --max_pronunciation_len=11 --max_spelling_len=3 --batch_size 1
```

Latest model from `/tmp/logdir/train/ckpt-00037150`.
