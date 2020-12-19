import torch.nn as nn
from Models.DCGAN_Model.Discriminator import Discriminator
from Models.DCGAN_Model.Generator import Generator
from torch import optim


class dc_gan(nn.Module):
    def __init__(self, args):
        super(dc_gan, self).__init__()
        self.args = args

        self.Generator = Generator(self.args)
        self.Discriminator = Discriminator(self.args)

        self.optimizer_generator = optim.Adam(self.Generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizer_discriminator = optim.Adam(self.Discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

        self.adversarial_loss = nn.BCELoss()

        self.valid = None
        self.fake = None
        self.real_loss = None

    def learn_discriminator(self, inputs, noise, real_labels, fake_labels):
        output = self.Discriminator(inputs)
        real_loss = self.adversarial_loss(output, real_labels)

        fake = self.Generator(noise)
        discriminator_result = self.Discriminator(fake.detach())
        fake_loss = self.adversarial_loss(discriminator_result, fake_labels)

        loss = real_loss + fake_loss

        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()

        return loss.item(), fake

    def learn_generator(self, noise, label):
        fake = self.Generator(noise)
        discriminator_result = self.Discriminator(fake)
        loss = self.adversarial_loss(discriminator_result, label)

        self.optimizer_generator.zero_grad()
        loss.backward()
        self.optimizer_generator.step()

        return loss.item()


