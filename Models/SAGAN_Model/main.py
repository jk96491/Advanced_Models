import torch
from Models.SAGAN_Model.Parser_args import parse_Arg
from Models.SAGAN_Model.SAGAN import sa_gan
from torchvision.utils import save_image
from Utils import CIFARLoadData
from Utils import get_device

args = parse_Arg()
device1 = get_device('cuda:0')
device2 = get_device('cuda:1')

train_loader = CIFARLoadData(args.batch_size, True, True)

model = sa_gan(args, [device1, device2])

for epoch in range(args.n_epochs):
    for i, data in enumerate(train_loader):
        real_images, _ = data
        current_batch_size = real_images.size(0)

        inputs = real_images.clone().to(device1)
        noise = torch.zeros(current_batch_size, args.noise_dim, 1, 1).normal_(0, 1)

        real_labels = torch.ones(current_batch_size, 1).detach()
        fake_labels = torch.zeros(current_batch_size, 1).detach()

        discriminator_loss, generator_image = model.learn_discriminator(inputs, noise, real_labels, fake_labels)
        generator_loss = model.learn_generator(noise, real_labels)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [Discriminator_loss: %f] [Generator_loss: %f]"
            % (epoch + 1, args.n_epochs, i + 1, len(train_loader), discriminator_loss, generator_loss)
        )

        batches_done = epoch * len(train_loader) + i
        if batches_done % args.sample_interval == 0:
            temp = generator_image
            save_image(generator_image, "images/%d.png" % batches_done, nrow=args.nrow, normalize=True)






