```shell
python neighborhood/data/create_neighborhood.py --output_dir /tmp/tmp --max_rand_len 10 --language_group luangthongkumkaren --lang all --pairwise_algo lingpy --random_target_algo markov --logtostderr --task_data_dir ~/projects/ST2022/data-surprise  --num_duplicates 100
```

```shell
python neighborhood/model/trainer.py  --run_locally=cpu --model=feature_neighborhood_model_config.TransformerWithNeighborsTiny --input_symbols /tmp/tmp/luangthongkumkaren.syms --output_symbols /tmp/tmp/luangthongkumkaren.syms --feature_neighborhood_train_path tfrecord:/tmp/tmp/luangthongkumkaren_train.tfrecords --feature_neighborhood_dev_path tfrecord:/tmp/tmp/luangthongkumkaren_test.tfrecords --feature_neighborhood_test_path tfrecord:/tmp/tmp/luangthongkumkaren_test.tfrecords --learning_rate=0.001 --max_neighbors=7 --max_pronunciation_len=8 --max_spelling_len=12 --logdir /tmp/logdir/
```

```shell
python neighborhood/model/decoder.py --feature_neighborhood_test_path=tfrecord:/tmp/tmp/luangthongkumkaren_test.tfrecords --input_symbols /tmp/tmp/luangthongkumkaren.syms --output_symbols /tmp/tmp/luangthongkumkaren.syms --ckpt /usr/local/google/sweet/sigtyp2022/luangthongkumkaren/train/ --model=TransformerWithNeighborsTiny --decode_dir /tmp/tmp/decode_dir/ --split_output_on_space --max_neighbors=7 --max_pronunciation_len=8 --max_spelling_len=12 --batch_size 1
```
