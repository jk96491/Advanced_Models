import argparse


def parse_Arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--layers", type=int, default=[3, 4, 6, 3])
    args = parser.parse_args()
    return args