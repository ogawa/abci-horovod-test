# ABCI Horovod test

## Preparation

```
$ git clone https://github.com/ogawa/abci-horovod-test
$ cd abci-horovod-test
$ git submodule init
$ git submodule update
```

## Run

```
$ qsub -g <group_id> cuda10.0-openmpi2.1.6.sh pytorch_mnist
$ qsub -g <group_id> cuda10.0-openmpi2.1.6.sh tensorflow_mnist
$ qsub -g <group_id> cuda10.0-openmpi2.1.6.sh tensorflow_word2vec
$ qsub -g <group_id> cuda10.0-openmpi2.1.6.sh keras_mnist
```
