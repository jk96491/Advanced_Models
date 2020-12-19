import torch
import torch.nn as nn
from Models.CGAN_Model.Discriminator import Discriminator
from Models.CGAN_Model.Generator import Generator
from torch import optim
import numpy as np
from torch.autograd import Variable


class cgan(nn.Module):
    def __init__(self, image_shape, args, device):
        super(cgan, self).__init__()
        self.image_shape = image_shape
        self.args = args
        self.device = device

        self.Generator = Generator(self.image_shape, self.args)
        self.Discriminator = Discriminator(self.image_shape)

        self.optimizer_generator = optim.Adam(self.Generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizer_discriminator = optim.Adam(self.Discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

        self.adversarial_loss = nn.BCELoss()

        self.valid = None
        self.fake = None

        self.to(device)

    def learn_generator(self, image, labels):
        self.valid = Variable(torch.FloatTensor(image.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
        self.fake = Variable(torch.FloatTensor(image.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)

        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (image.shape[0],  self.args.latent_dim)))).to(self.device)

        generator_images = self.Generator(z, labels)
        discriminator_result = self.Discriminator(generator_images, labels)
        loss = self.adversarial_loss(discriminator_result, self.valid)

        self.optimizer_generator.zero_grad()
        loss.backward()
        self.optimizer_generator.step()

        return loss.item(), generator_images

    def learn_discriminator(self, real_images, generator_images, labels):
        real_loss = self.adversarial_loss(self.Discriminator(real_images, labels), self.valid)
        fake_loss = self.adversarial_loss(self.Discriminator(generator_images.detach(), labels), self.fake)

        loss = (real_loss + fake_loss) / 2

        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()

        return loss.item()


