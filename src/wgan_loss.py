import torch
from torchgan.losses import DiscriminatorLoss, GeneratorLoss
import torch.autograd as autograd
from betaVAE import betaVAE
from sklearn.preprocessing import StandardScaler

def reduce_vae(x, reduction=None):
    r"""Applies reduction on a torch.Tensor.
    Args:
        x (torch.Tensor): The tensor on which reduction is to be applied.
        reduction (str, optional): The reduction to be applied. If ``mean`` the  mean value of the
            Tensor is returned. If ``sum`` the elements of the Tensor will be summed. If none of the
            above then the Tensor is returning without any change.
    Returns:
        As per the above ``reduction`` convention.
    """
    if reduction == "mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:
        return x

def wasserstein_generator_loss_vae(fgz, reduction="mean"):
    return reduce_vae(-1.0 * fgz, reduction='mean')


def wasserstein_discriminator_loss_vae(fx, fgz, reduction="mean"):
    return reduce_vae(fgz - fx, reduction='mean')


def wasserstein_gradient_penalty_vae(interpolate, d_interpolate, reduction="mean"):
    grad_outputs = torch.ones_like(d_interpolate)
    gradients = autograd.grad(
        outputs=d_interpolate,
        inputs=interpolate,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = (gradients.norm(2) - 1) ** 2
    return reduce_vae(gradient_penalty, reduction='mean')


class WassersteinGeneratorLossVAE(GeneratorLoss):
    r"""Wasserstein GAN generator loss from
    `"Wasserstein GAN by Arjovsky et. al." <https://arxiv.org/abs/1701.07875>`_ paper
    The loss can be described as:
    .. math:: L(G) = -f(G(z))
    where
    - :math:`G` : Generator
    - :math:`f` : Critic/Discriminator
    - :math:`z` : A sample from the noise prior
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """
    def __init__(self, checkpoint, rna_features, beta=0.005):
        super(WassersteinGeneratorLossVAE, self).__init__(
            checkpoint, rna_features
        )
        self.betavae = betaVAE(rna_features, 2048, [6000, 4000, 2048], [4000, 6000], beta=beta)
        self.betavae.load_state_dict(torch.load(checkpoint))
        self.betavae.eval()

    def forward(self, fgz):
        r"""Computes the loss for the given input.
        Args:
            dgz (torch.Tensor) : Output of the Discriminator with generated data. It must have the
                                 dimensions (N, \*) where \* means any number of additional
                                 dimensions.
        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return wasserstein_generator_loss_vae(fgz, self.reduction)

    def train_ops(self,
        generator,
        discriminator,
        optimizer_generator,
        device,
        batch_size,
        real_inputs,
        labels=None,
        ):
        if labels is None and generator.label_type == "required":
                raise Exception("GAN model requires labels for training")

        batch_size = real_inputs['image'].size(0)
        gene_coding = real_inputs['rna_data']
        self.betavae = self.betavae.to(device)
        z, _, _ = self.betavae.encode(gene_coding.to(device))
        z = z.to(device)
        # generate noise in range
        noise = torch.FloatTensor(batch_size, generator.encoding_dims).uniform_(-0.3, 0.3)
        noise = noise.to(device)
        #noise = torch.randn(
        #    batch_size, generator.encoding_dims, device=device
        #)
        noise = noise + z
        noise = (noise - torch.mean(noise, dim=0)) / torch.std(noise, dim=0)
        optimizer_generator.zero_grad()
        if generator.label_type == "generated":
            label_gen = torch.randint(
                0, generator.num_classes, (batch_size,), device=device
            )
        if generator.label_type == "none":
            fake = generator(noise)
        elif generator.label_type == "required":
            fake = generator(noise, labels)
        elif generator.label_type == "generated":
            fake = generator(noise, label_gen)
        if discriminator.label_type == "none":
            dgz = discriminator(fake)
        else:
            if generator.label_type == "generated":
                dgz = discriminator(fake, label_gen)
            else:
                dgz = discriminator(fake, labels)
        loss = self.forward(dgz)
        loss.backward()
        optimizer_generator.step()
        # NOTE(avik-pal): This will error if reduction is is 'none'
        return loss.item()

class WassersteinDiscriminatorLossVAE(DiscriminatorLoss):
    r"""Wasserstein GAN generator loss from
    `"Wasserstein GAN by Arjovsky et. al." <https://arxiv.org/abs/1701.07875>`_ paper
    The loss can be described as:
    .. math:: L(D) = f(G(z)) - f(x)
    where
    - :math:`G` : Generator
    - :math:`f` : Critic/Discriminator
    - :math:`x` : A sample from the data distribution
    - :math:`z` : A sample from the noise prior
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        clip (tuple, optional): Tuple that specifies the maximum and minimum parameter
            clamping to be applied, as per the original version of the Wasserstein loss
            without Gradient Penalty.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """

    def __init__(self, checkpoint, rna_features, beta=0.005, reduction="mean", clip=None, override_train_ops=None):
        super(WassersteinDiscriminatorLossVAE, self).__init__(
            checkpoint, rna_features
        )
        if (isinstance(clip, tuple) or isinstance(clip, list)) and len(
            clip
        ) > 1:
            self.clip = clip
        else:
            self.clip = None
        
        self.betavae = betaVAE(rna_features, 2048, [6000, 4000, 2048], [4000, 6000], beta=beta)
        self.betavae.load_state_dict(torch.load(checkpoint))
        self.betavae.eval()

    def forward(self, fx, fgz):
        r"""Computes the loss for the given input.
        Args:
            fx (torch.Tensor) : Output of the Discriminator with real data. It must have the
                                dimensions (N, \*) where \* means any number of additional
                                dimensions.
            fgz (torch.Tensor) : Output of the Discriminator with generated data. It must have the
                                 dimensions (N, \*) where \* means any number of additional
                                 dimensions.
        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return wasserstein_discriminator_loss_vae(fx, fgz, self.reduction)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by wasserstein discriminator loss.
        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:
        1. Clamp the discriminator parameters to satisfy :math:`lipschitz\ condition`
        2. :math:`fake = generator(noise)`
        3. :math:`value_1 = discriminator(fake)`
        4. :math:`value_2 = discriminator(real)`
        5. :math:`loss = loss\_function(value_1, value_2)`
        6. Backpropagate by computing :math:`\nabla loss`
        7. Run a step of the optimizer for discriminator
        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            labels (torch.Tensor, optional): Labels for the data.
        Returns:
            Scalar value of the loss.
        """
        
        if self.clip is not None:
            for p in discriminator.parameters():
                p.data.clamp_(self.clip[0], self.clip[1])
        if labels is None and (
            generator.label_type == "required"
            or discriminator.label_type == "required"
        ):
            raise Exception("GAN model requires labels for training")
        batch_size = real_inputs['image'].size(0)
        gene_coding = real_inputs['rna_data']
        self.betavae = self.betavae.to(device)
        z, _, _ = self.betavae.encode(gene_coding.to(device))
        z = z.to(device)
        # generate noise in range
        noise = torch.FloatTensor(batch_size, generator.encoding_dims).uniform_(-0.3, 0.3)
        noise = noise.to(device)
        #noise = torch.randn(
        #    batch_size, generator.encoding_dims, device=device
        #)
        noise = noise + z
        noise = (noise - torch.mean(noise, dim=0)) / torch.std(noise, dim=0)
        if generator.label_type == "generated":
            label_gen = torch.randint(
                0, generator.num_classes, (batch_size,), device=device
            )
        real_image = real_inputs['image'].to(device)
        optimizer_discriminator.zero_grad()
        if discriminator.label_type == "none":
            dx = discriminator(real_image)
        elif discriminator.label_type == "required":
            dx = discriminator(real_image, labels)
        else:
            dx = discriminator(real_image, label_gen)
        if generator.label_type == "none":
            fake = generator(noise)
        elif generator.label_type == "required":
            fake = generator(noise, labels)
        else:
            fake = generator(noise, label_gen)
        if discriminator.label_type == "none":
            dgz = discriminator(fake.detach())
        else:
            if generator.label_type == "generated":
                dgz = discriminator(fake.detach(), label_gen)
            else:
                dgz = discriminator(fake.detach(), labels)
        loss = self.forward(dx, dgz)
        loss.backward()
        optimizer_discriminator.step()
        # NOTE(avik-pal): This will error if reduction is is 'none'
        return loss.item()


class WassersteinGradientPenaltyVAE(DiscriminatorLoss):
    r"""Gradient Penalty for the Improved Wasserstein GAN discriminator from
    `"Improved Training of Wasserstein GANs
    by Gulrajani et. al." <https://arxiv.org/abs/1704.00028>`_ paper
    The gradient penalty is calculated as:
    .. math: \lambda \times (||\nabla(D(x))||_2 - 1)^2
    The gradient being taken with respect to x
    where
    - :math:`G` : Generator
    - :math:`D` : Disrciminator/Critic
    - :math:`\lambda` : Scaling hyperparameter
    - :math:`x` : Interpolation term for the gradient penalty
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        lambd (float,optional): Hyperparameter lambda for scaling the gradient penalty.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """

    def __init__(self, checkpoint, rna_features, reduction="mean", lambd=10.0, override_train_ops=None, beta=0.005):
        super(WassersteinGradientPenaltyVAE, self).__init__(
            checkpoint, rna_features
        )
        self.lambd = lambd
        self.override_train_ops = override_train_ops
        self.betavae = betaVAE(rna_features, 2048, [6000, 4000, 2048], [4000, 6000], beta=beta)
        self.betavae.load_state_dict(torch.load(checkpoint))
        self.betavae.eval()
    def forward(self, interpolate, d_interpolate):
        r"""Computes the loss for the given input.
        Args:
            interpolate (torch.Tensor) : It must have the dimensions (N, \*) where
                                         \* means any number of additional dimensions.
            d_interpolate (torch.Tensor) : Output of the ``discriminator`` with ``interpolate``
                                           as the input. It must have the dimensions (N, \*)
                                           where \* means any number of additional dimensions.
        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        # TODO(Aniket1998): Check for performance bottlenecks
        # If found, write the backprop yourself instead of
        # relying on autograd
        return wasserstein_gradient_penalty_vae(
            interpolate, d_interpolate, self.reduction
        )

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by the Wasserstein Gradient Penalty.
        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:
        1. :math:`fake = generator(noise)`
        2. :math:`interpolate = \epsilon \times real + (1 - \epsilon) \times fake`
        3. :math:`d\_interpolate = discriminator(interpolate)`
        4. :math:`loss = \lambda loss\_function(interpolate, d\_interpolate)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for discriminator
        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.
        Returns:
            Scalar value of the loss.
        """
        
        if labels is None and (
            generator.label_type == "required"
            or discriminator.label_type == "required"
        ):
            raise Exception("GAN model requires labels for training")
        batch_size = real_inputs['image'].size(0)
        gene_coding = real_inputs['rna_data']
        self.betavae = self.betavae.to(device)
        z, _, _ = self.betavae.encode(gene_coding.to(device))
        z = z.detach().to(device)
        # generate noise in range
        noise = torch.FloatTensor(batch_size, generator.encoding_dims).uniform_(-0.3, 0.3)
        noise = noise.to(device)
        #noise = torch.randn(
        #    batch_size, generator.encoding_dims, device=device
        #)
        noise = noise + z
        noise = (noise - torch.mean(noise, dim=0)) / torch.std(noise, dim=0)
        real_image = real_inputs['image'].to(device)
        if generator.label_type == "generated":
            label_gen = torch.randint(
                0, generator.num_classes, (batch_size,), device=device
            )
        optimizer_discriminator.zero_grad()
        if generator.label_type == "none":
            fake = generator(noise)
        elif generator.label_type == "required":
            fake = generator(noise, labels)
        else:
            fake = generator(noise, label_gen)
        eps = torch.rand(1).item()
        interpolate = eps * real_image + (1 - eps) * fake
        if discriminator.label_type == "none":
            d_interpolate = discriminator(interpolate)
        else:
            if generator.label_type == "generated":
                d_interpolate = discriminator(interpolate, label_gen)
            else:
                d_interpolate = discriminator(interpolate, labels)
        loss = self.forward(interpolate, d_interpolate)
        weighted_loss = self.lambd * loss
        weighted_loss.backward()
        optimizer_discriminator.step()
        return loss.item()