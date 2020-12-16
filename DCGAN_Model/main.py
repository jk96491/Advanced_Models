import torch
from DCGAN_Model.Parser_args import parse_Arg
from torch.autograd import Variable
from DCGAN_Model.DCGAN import dc_gan
from torchvision.utils import save_image
from Utils import CIFARLoadData


args = parse_Arg()

train_loader = CIFARLoadData(args.batch_size, True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = dc_gan(args).to(device)

inputs = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size).to(device)
noise = torch.FloatTensor(args.batch_size, args.noise_dim, 1, 1).to(device)
label = torch.FloatTensor(args.batch_size).to(device)

for epoch in range(args.n_epochs):
    for i, data in enumerate(train_loader):
        real_images, _ = data

        real_labels = label.resize_(args.batch_size).fill_(1)
        fake_labels = label.resize_(args.batch_size).fill_(0)

        inputs = Variable(inputs.resize_as_(real_images).copy_(real_images)).to(device)
        noise.resize_(args.batch_size, args.noise_dim, 1, 1).normal_(0, 1)
        noise = Variable(noise)

        discriminator_loss, generator_image = model.learn_discriminator(inputs, noise, real_labels, fake_labels)
        generator_loss = model.learn_generator(real_labels)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [Discriminator_loss: %f] [Generator_loss: %f]"
            % (epoch, args.n_epochs, i, len(train_loader), discriminator_loss, generator_loss)
        )

        batches_done = epoch * len(train_loader) + i
        if batches_done % args.sample_interval == 0:
            temp = generator_image
            save_image(generator_image, "images/%d.png" % batches_done, nrow=8, normalize=True)





