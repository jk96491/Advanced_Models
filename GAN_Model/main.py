import torch
from GAN_Model.Parser_args import parse_Arg
from Utils import CIFARLoadData
from torch.autograd import Variable
from GAN_Model.GAN import gan
from torchvision.utils import save_image
from Utils import get_device

args = parse_Arg()
image_shape = (args.channels, args.image_size, args.image_size)
data_loader = CIFARLoadData(args.batch_size, True)

device = get_device()

model = gan(image_shape, args, device)

for epoch in range(args.n_epochs):
    for i, (images, _) in enumerate(data_loader):
        real_images = Variable(images.type(torch.FloatTensor)).to(device)

        generator_loss, generator_image = model.learn_generator(images)
        discriminator_loss = model.learn_discriminator(real_images, generator_image)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [Discriminator_loss: %f] [Generator_loss: %f]"
            % (epoch, args.n_epochs, i, len(data_loader), discriminator_loss, generator_loss)
        )

        batches_done = epoch * len(data_loader) + i
        if batches_done % args.sample_interval == 0:
            save_image(generator_image.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)





