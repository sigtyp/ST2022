# `bodtkhobwa`

## Data generation

```shell
python neighbors/data/create_neighbors.py --output_dir /tmp/tmp --max_rand_len 10 --language_group bodtkhobwa --lang all --pairwise_algo lingpy --random_target_algo markov --logtostderr --task_data_dir ~/projects/ST2022/data-surprise  --num_duplicates 50
```

## Training

```shell
python neighbors/model/trainer.py  --run_locally=cpu --model=feature_neighbors_model_config.TransformerWithNeighborsTiny --input_symbols /tmp/tmp/bodtkhobwa.syms --output_symbols /tmp/tmp/bodtkhobwa.syms --feature_neighbors_train_path tfrecord:/tmp/tmp/bodtkhobwa_train.tfrecords --feature_neighbors_dev_path tfrecord:/tmp/tmp/bodtkhobwa_test.tfrecords --feature_neighbors_test_path tfrecord:/tmp/tmp/bodtkhobwa_test.tfrecords --learning_rate=0.001 --max_neighbors=7 --max_pronunciation_len=6 --max_spelling_len=9 --logdir /tmp/logdir/
```

## Inference

```shell
python neighbors/model/decoder.py --feature_neighbors_test_path=tfrecord:/tmp/tmp/bodtkhobwa_test.tfrecords --input_symbols /tmp/tmp/bodtkhobwa.syms --output_symbols /tmp/tmp/bodtkhobwa.syms --ckpt /tmp/logdir/train --model=TransformerWithNeighborsTiny --decode_dir /tmp/tmp/decode_dir/ --split_output_on_space --max_neighbors=7 --max_pronunciation_len=6 --max_spelling_len=9 --batch_size 1
```