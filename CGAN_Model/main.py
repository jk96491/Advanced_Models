import torch
from CGAN_Model.Parser_args import parse_Arg
from Utils import CIFARLoadData
from torch.autograd import Variable
from CGAN_Model.CGAN import cgan
from torchvision.utils import save_image
from Utils import get_device

args = parse_Arg()
image_shape = (args.channels, args.image_size, args.image_size)
data_loader = CIFARLoadData(args.batch_size, True, True)

device = get_device("cuda:1")

model = cgan(image_shape, args, device)

for epoch in range(args.n_epochs):
    for i, (images, label) in enumerate(data_loader):
        real_images = Variable(images.type(torch.FloatTensor)).to(device)
        label = Variable(label.type(torch.LongTensor)).to(device)

        y_label_ = torch.zeros(images.size(0), 10)
        y_label_.scatter_(1, label.view(images.size(0), 1), 1)
        labels = torch.FloatTensor(y_label_)

        generator_loss, generator_image = model.learn_generator(images, labels)
        discriminator_loss = model.learn_discriminator(real_images, generator_image, labels)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [Discriminator_loss: %f] [Generator_loss: %f]"
            % (epoch + 1, args.n_epochs, i + 1, len(data_loader), discriminator_loss, generator_loss)
        )

        batches_done = epoch * len(data_loader) + i
        if batches_done % args.sample_interval == 0:
            save_image(generator_image, "images/%d.png" % batches_done, nrow=args.nrow, normalize=True)





