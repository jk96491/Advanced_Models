import torch
import torch.nn as nn
from Models.SAGAN_Model.Discriminator import Discriminator
from Models.SAGAN_Model.Generator import Generator
from torch import optim


class sa_gan(nn.Module):
    def __init__(self, args, device):
        super(sa_gan, self).__init__()
        self.args = args

        self.device_generator = device[0]
        self.device_discriminator = device[1]

        self.Generator = Generator(self.args).to(self.device_generator)
        self.Discriminator = Discriminator(self.args).to(self.device_discriminator)

        self.optimizer_generator = optim.Adam(self.Generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizer_discriminator = optim.Adam(self.Discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

        self.adversarial_loss = nn.BCELoss()

        self.valid = None
        self.fake = None
        self.real_loss = None

    def learn_discriminator(self, inputs, noise, real_labels, fake_labels):
        output = self.Discriminator(inputs)
        real_loss = self.adversarial_loss(output, real_labels.to(self.device_discriminator))

        fake = self.Generator(noise)
        discriminator_result = self.Discriminator(fake.detach().to(self.device_discriminator))
        fake_loss = self.adversarial_loss(discriminator_result, fake_labels.to(self.device_discriminator))

        loss = real_loss + fake_loss

        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()

        return loss.item(), fake

    def learn_generator(self, noise, label):
        fake = self.Generator(noise.to(self.device_generator))
        discriminator_result = self.Discriminator(fake.to(self.device_discriminator))
        loss = self.adversarial_loss(discriminator_result, label.to(self.device_discriminator))

        self.optimizer_generator.zero_grad()
        loss.backward()
        self.optimizer_generator.step()

        return loss.item()





