# BlockShuffleTest
BlockShuffle，就是在训练过程中使用分块打乱替代随机打乱的一种方法，即将原始数据按照数据长度进行排序，然后进行batch划分，在对batch训练进行打乱。这样操作，可以减少数据padding长度，缩短训练时长。
