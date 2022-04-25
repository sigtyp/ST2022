# `allenbai`

## Data generation

```shell
python neighbors/data/create_neighborhood.py --output_dir /tmp/tmp --max_rand_len 10 --language_group allenbai --lang all --pairwise_algo lingpy --random_target_algo markov --logtostderr --task_data_dir ~/projects/ST2022/data  --num_duplicates 100
```

## Training

```shell
python neighbors/model/trainer.py  --run_locally=cpu --model=feature_neighborhood_model_config.TransformerWithNeighborsTiny --input_symbols /tmp/tmp/allenbai.syms --output_symbols /tmp/tmp/allenbai.syms --feature_neighborhood_train_path tfrecord:/tmp/tmp/allenbai_train.tfrecords --feature_neighborhood_dev_path tfrecord:/tmp/tmp/allenbai_test.tfrecords --feature_neighborhood_test_path tfrecord:/tmp/tmp/allenbai_test.tfrecords --learning_rate=0.001 --max_neighbors=8 --max_pronunciation_len=5 --max_spelling_len=11 --logdir /tmp/logdir/
```

## Inference

```shell
python neighbors/model/decoder.py --feature_neighborhood_test_path=tfrecord:/tmp/tmp/allenbai_test.tfrecords --input_symbols /tmp/tmp/allenbai.syms --output_symbols /tmp/tmp/allenbai.syms --ckpt /tmp/logdir/train --model=TransformerWithNeighborsTiny --decode_dir /tmp/tmp/decode_dir/ --split_output_on_space --max_neighbors=8 --max_pronunciation_len=5 --max_spelling_len=11 --batch_size 1
```

Latest model from `/tmp/logdir/train/ckpt-00017208`.

