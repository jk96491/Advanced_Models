import torch
import torch.nn as nn
from GAN_Model.Discriminator import Discriminator
from GAN_Model.Generator import Generator
from torch import optim
import numpy as np
from torch.autograd import Variable


class gan(nn.Module):
    def __init__(self, image_shape, args):
        super(gan, self).__init__()
        self.image_shape = image_shape
        self.args = args

        self.Generator = Generator(self.image_shape, self.args)
        self.Discriminator = Discriminator(self.image_shape)

        self.optimizer_generator = optim.Adam(self.Generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizer_discriminator = optim.Adam(self.Discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

        self.adversarial_loss = nn.BCELoss()

        self.valid = None
        self.fake = None

    def learn_generator(self, image):
        self.valid = Variable(torch.FloatTensor(image.size(0), 1).fill_(1.0), requires_grad=False)
        self.fake = Variable(torch.FloatTensor(image.size(0), 1).fill_(0.0), requires_grad=False)

        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (image.shape[0],  self.args.latent_dim))))

        generator_images = self.Generator(z)
        discriminator_result = self.Discriminator(generator_images)
        loss = self.adversarial_loss(discriminator_result, self.valid)

        self.optimizer_generator.zero_grad()
        loss.backward()
        self.optimizer_generator.step()

        return loss.item(), generator_images

    def learn_discriminator(self, real_images, generator_images):
        real_loss = self.adversarial_loss(self.Discriminator(real_images), self.valid)
        fake_loss = self.adversarial_loss(self.Discriminator(generator_images.detach()), self.fake)

        loss = (real_loss + fake_loss) / 2

        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()

        return loss.item()


