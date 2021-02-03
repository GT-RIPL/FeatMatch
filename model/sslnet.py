from torch import nn


class SSLNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.0, negative_slope=0.0):
        super(SSLNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.negative_slope = negative_slope

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out.view(out.size(0), -1))

        return out
