from Utils import MnistLoadData
from Utils import CIFARLoadData
from Models.VAE_Model.Parser_args import parse_Arg
from Models.VAE_Model.VAE import vae
from torch.autograd import Variable
from torchvision.utils import save_image
from Utils import get_device

args = parse_Arg()
image_shape = (args.channels, args.image_size, args.image_size)
data_loader = CIFARLoadData(args.batch_size, True, True)

device = get_device()

model = vae(args, device)

for epoch in range(args.n_epochs):
    for i, data in enumerate(data_loader, 0):
        inputs, _ = data
        current_batch_size = inputs.size(0)
        inputs = inputs.view(current_batch_size, args.input_dim * args.channels).to(device)

        loss = model.learn(inputs)

        print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]]" % (epoch + 1, args.n_epochs, i + 1, len(data_loader), loss))

        batches_done = epoch * len(data_loader) + i
        if batches_done % args.sample_interval == 0:
            output = model(inputs).data.reshape(args.batch_size, args.channels, args.image_size, args.image_size)
            save_image(output, "images/%d.png" % batches_done, nrow=args.nrow, normalize=True)




