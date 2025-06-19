import torch
import torch.nn as nn
from .convlstm import ConvLSTM
from .differentiable_reco import DifferentiableReconstruction
from .ssl import SelfSupervisedLearning
from .adversarial import AdversarialDiscriminator


class Freehand3DUSReconstruction(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, volume_size):
        super(Freehand3DUSReconstruction, self).__init__()
        self.convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers)
        self.differentiable_reco = DifferentiableReconstruction(volume_size)
        self.ssl = SelfSupervisedLearning()
        self.discriminator = AdversarialDiscriminator(volume_size)

    def forward(self, input_seq):
        transforms = self.convlstm(input_seq)
        volume = self.differentiable_reco(input_seq, transforms)
        return transforms, volume

    def calculate_ssl_loss(self, input_seq, transforms):
        return self.ssl(input_seq, transforms, self.differentiable_reco)

    def calculate_adversarial_loss(self, generated_volume, real_volume):
        fake_pred = self.discriminator(generated_volume)
        real_pred = self.discriminator(real_volume)

        adv_loss = -torch.mean(torch.log(fake_pred))
        disc_loss = -torch.mean(torch.log(real_pred)) - torch.mean(torch.log(1 - fake_pred))

        return adv_loss, disc_loss