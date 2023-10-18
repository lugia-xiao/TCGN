from torchvision.ops import StochasticDepth
import torch.nn as nn
import torch


class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)),
                                  requires_grad=True)

    def forward(self, x):
        return self.gamma[None, ..., None, None] * x


class BottleNeckBlock(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            expansion: int = 4,
            drop_p: float = .0,
            layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        self.flag=True
        expanded_features = in_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(
                in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv2d(expanded_features, in_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, in_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")
        if in_features!=out_features:
            self.proj=nn.Conv2d(in_features, out_features, kernel_size=1)
            self.flag=False

    def forward(self, x):
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        if self.flag==False:
            x=self.proj(x)
        return x


class ConvNexStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
            ),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )

def get_size():
    model=ConvNexStage(46,92,1)
    #model=models.densenet121()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    print(param_size, param_sum, buffer_size, buffer_sum, all_size)

if __name__=="__main__":
    get_size()
    model=ConvNexStage(46,92,1).cuda()
    x=torch.randn((2,46,56,56)).cuda()
    out=model(x)
    print(out.shape)