from torch import nn
import torch

class GANLoss(nn.Module):
    """ Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) -- the type of GAN objectives: vanilla | lsgan | wgangp
            target_real_label (bool) -- label for a real image
            target_fake_label (bool) -- label for a fake image

        Note: Do ot use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError("gan mode {} not implemented".format(gan_mode))

    def getTargetTensor(self, prediction, target_is_real):
        """ Create label tensors with the same size as the input.

        Parameters: 
            prediction (tensor)   -- typically the prediction from a discriminator
            target_is_real (bool) -- if the ground truth label if for real images or fake images

        Returns: 
            A label tensor filled with ground truth label, and with the size of the input.
        """
        
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """ Caluculate loss given Discriminator's output (prediction) and ground truth label.

        Parameters: 
            prediction (tensor)   -- typically the prediction from a discriminator
            target_is_real (bool) -- if the ground truth label if for real images or fake images

        Returns: 
            the calculated loss.
        """

        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.getTargetTensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()

        return loss
            

# Test
if __name__ == "__main__":
    loss_v  = GANLoss("vanilla")
    loss_ls = GANLoss("lsgan")
    loss_w  = GANLoss("wgangp")

    def testClass(class_, input_tensor):
        t = class_(input_tensor, True)
        f = class_(input_tensor, False)

        print("Input tensor: ", input_tensor)
        print("When True: ", t)
        print("When False: ", f)

    input_tensor = torch.tensor([0.2, 0.3, 0.4, 0.1])
    testClass(loss_v, input_tensor)
    testClass(loss_ls, input_tensor)
    testClass(loss_ls, input_tensor)


