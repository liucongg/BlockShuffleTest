# BlockShuffleTest
BlockShuffle，就是在训练过程中使用分块打乱替代随机打乱的一种方法，即将原始数据按照数据长度进行排序，然后进行batch划分，在对batch训练进行打乱。这样操作，可以减少数据padding长度，缩短训练时长。

# Run
```bash
python train.py --device=3 --train_file_path=./data/train.json --dev_file_path=./data/test.json --data_dir=./data/ --num_train_epochs=2 --train_batch_size=32 --test_batch_size=32 --learning_rate=5e-5 --warmup_proportion=0.1 --adam_epsilon=1e-8 --save_model_steps=12 --logging_steps=5 --gradient_accumulation_steps=1 --max_grad_norm=1.0 --output_dir=output_dir --seed=2020 --max_len=256 --num_labels=6
```