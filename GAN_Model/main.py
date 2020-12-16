from GAN_Model.Parser_args import parse_Arg
from Utils import MnistLoadData


args = parse_Arg()

image_shape = (args.channels, args.image_size, args.image_size)

data_loader = MnistLoadData(args.image_size, args.batch_size)