import torch
from DCGAN_Model.Parser_args import parse_Arg
from torch.autograd import Variable
from DCGAN_Model.DCGAN import dc_gan
from torchvision.utils import save_image
from Utils import CIFARLoadData


args = parse_Arg()

train_loader = CIFARLoadData(args.batch_size, True)

model = dc_gan(args)

inputs = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
noise = torch.FloatTensor(args.batch_size, args.noise_dim, 1, 1)
label = torch.FloatTensor(args.batch_size)

real_label = 1
fake_label = 0

for epoch in range(args.n_epochs):
    for i, data in enumerate(train_loader):
        real_images, _ = data

        inputs = Variable(inputs.resize_as_(real_images).copy_(real_images))
        noise.resize_(args.batch_size, args.noise_dim, 1, 1).normal_(0, 1)
        noise = Variable(noise)
        label = Variable(label.fill_(fake_label))

        discriminator_loss, generator_image = model.learn_discriminator(inputs, noise, label)
        generator_loss = model.learn_generator(label)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [Discriminator_loss: %f] [Generator_loss: %f]"
            % (epoch, args.n_epochs, i, len(train_loader), discriminator_loss, generator_loss)
        )

        batches_done = epoch * len(train_loader) + i
        if batches_done % args.sample_interval == 0:
            temp = generator_image
            save_image(generator_image, "images/%d.png" % batches_done, nrow=8, normalize=True)





