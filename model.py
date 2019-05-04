import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 4 input channels
        self.conv11 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv21 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv31 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv32 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv41 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv42 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv51 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv52 = nn.Conv2d(512, 512, 3, padding=1)

        # in_channels, out_channels, kernel-size(need check...)
        self.deconv5 = nn.ConvTranspose2d(512, 256, (2, 2), (2, 2))

        self.conv61 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv62 = nn.Conv2d(256, 256, 3, padding=1)

        self.deconv6 = nn.ConvTranspose2d(256, 128, (2, 2), (2, 2))

        self.conv71 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv72 = nn.Conv2d(128, 128, 3, padding=1)

        self.deconv7 = nn.ConvTranspose2d(128, 64, (2, 2), (2, 2))

        self.conv81 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv82 = nn.Conv2d(64, 64, 3, padding=1)

        self.deconv8 = nn.ConvTranspose2d(64, 32, (2, 2), (2, 2))

        self.conv91 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv92 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv10 = nn.Conv2d(32, 12, 3, padding=1)

        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.leaky_relu(self.conv11(x))
        x_conv1 = F.leaky_relu(self.conv12(x))
        x = F.max_pool2d(x_conv1, (2, 2))

        x = F.leaky_relu(self.conv21(x))
        x_conv2 = F.leaky_relu(self.conv22(x))
        x = F.max_pool2d(x_conv2, (2, 2))

        x = F.leaky_relu(self.conv31(x))
        x_conv3 = F.leaky_relu(self.conv32(x))
        x = F.max_pool2d(x_conv3, (2, 2))

        x = F.leaky_relu(self.conv41(x))
        x_conv4 = F.leaky_relu(self.conv42(x))
        x = F.max_pool2d(x_conv4, (2, 2))

        x = F.leaky_relu(self.conv51(x))
        x = F.leaky_relu(self.conv52(x))

        deconv = self.deconv5(x)
        # print x_conv4.size(), deconv.size()
        up6 = self.concat(deconv, x_conv4, 256)
        # print "afterconcat:", up6.size()

        x = F.leaky_relu(self.conv61(up6))
        x = F.leaky_relu(self.conv62(x))

        deconv = self.deconv6(x)
        up7 = self.concat(deconv, x_conv3, 128)

        x = F.leaky_relu(self.conv71(up7))
        x = F.leaky_relu(self.conv72(x))

        deconv = self.deconv7(x)
        up8 = self.concat(deconv, x_conv2, 64)

        x = F.leaky_relu(self.conv81(up8))
        x = F.leaky_relu(self.conv82(x))

        deconv = self.deconv8(x)
        up9 = self.concat(deconv, x_conv1, 32)

        x = F.leaky_relu(self.conv91(up9))
        x = F.leaky_relu(self.conv92(x))

        x = self.conv10(x)

        out = self.pixelshuffle(x)

        return out

    def concat(self, deconv, x2, output_channels):
        deconv_out = torch.cat((deconv, x2), 1)
        return deconv_out


if __name__ == "__main__":
    net = ConvNet()
    print(net)