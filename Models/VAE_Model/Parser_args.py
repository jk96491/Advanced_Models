import argparse


def parse_Arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--input_dim", type=int, default=64*64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_dim2", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--image_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    args = parser.parse_args()
    return args