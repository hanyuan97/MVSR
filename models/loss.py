import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type='gan', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

class MVLoss(nn.Module):
    def __init__(self) -> None:
        super(MVLoss, self).__init__()
        self.loss = nn.MSELoss()
    def forward(self, lr, sr, hr):
        resize = F.interpolate(lr, size=(256, 448))
        mv2 = (torch.abs(resize[:,3:4,...]) + torch.abs(resize[:,3:4,...])) / 2
        mv3 = (torch.abs(resize[:,8:9,...]) + torch.abs(resize[:,8:9,...])) / 2
        sr_copy = sr.clone()
        hr_copy = hr.clone()
        # mv2_np = mv2[0].cpu().detach().numpy().transpose(1,2,0)*255
        # test = hr_copy[0,0:3,...] * (mv2[0]>0)
        # test = test.cpu().detach().numpy().transpose(1,2,0)
        # print(test.shape)
        # cv2.imwrite("test.png", test)
        # cv2.imwrite("mv.png", mv2_np)
        sr_copy[:,0:3,...] = sr_copy[:,0:3,...] * (mv2>0)
        sr_copy[:,3:6,...] = sr_copy[:,3:6,...] * (mv2>0)
        sr_copy[:,6:9,...] = sr_copy[:,6:9,...] * (mv3>0)
        hr_copy[:,0:3,...] = hr_copy[:,0:3,...] * (mv2>0)
        hr_copy[:,3:6,...] = hr_copy[:,3:6,...] * (mv2>0)
        hr_copy[:,6:9,...] = hr_copy[:,6:9,...] * (mv3>0)
        loss = self.loss(sr_copy, hr_copy)
        return loss

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output