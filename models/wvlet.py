import torch.nn as nn
import torch
import pywt
import numpy as np
from models.resnet_gated import FrequencyResnetGated


class WVlet(nn.Module):

    def __init__(self, in_channels=96, out_channel=96):
        super(WVlet, self).__init__()
        self.resnet_gated = FrequencyResnetGated(in_channels, out_channel)
        self.level = 2
        self.wvlet2_conv1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=0)

        self.wvlet3_conv1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1)
        self.wvlet3_conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1)

        self.wvletFusion = nn.Conv2d(in_channels=out_channel * 2, out_channels=240, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        coeffs_3d_list = self.wavelet_3d_transform(x)
        coeffs_x1_batch = []
        coeffs_x2_batch = []
        for coeffs_3d in coeffs_3d_list:
            coe_list = []
            [coe_list.extend(coeffs_3d[i][2][:]) for i in range(1, len(coeffs_3d))]  # 只是用高频特征
            coe_tensor_list = [torch.tensor(coe_list[i], dtype=torch.float32) for i in range(len(coe_list))]

            # coe_list = [coe_list[i].tolist() for i in range(len(coe_list))]
            coeffs_x1 = torch.cat(coe_tensor_list[:len(coe_list) // 2], axis=-1)  # img_h/4, img_w/4, 48
            coeffs_x2 = torch.cat(coe_tensor_list[len(coe_list) // 2:], axis=-1)  # img_h/2, img_w/2, 48
            coeffs_x1_batch.append(coeffs_x1)
            coeffs_x2_batch.append(coeffs_x2)
        # coeffs_x1_batch1 = torch.tensor([item.cpu().detach().numpy() for item in coeffs_x1_batch]).cuda()
        coeffs_x1_batch = torch.stack(coeffs_x1_batch).cuda()
        coeffs_x1_batch = coeffs_x1_batch.permute(0, 3, 1, 2)
        # coeffs_x2_batch = torch.tensor([item.cpu().detach().numpy() for item in coeffs_x2_batch]).cuda()
        coeffs_x2_batch = torch.stack(coeffs_x2_batch).cuda()
        coeffs_x2_batch = coeffs_x2_batch.permute(0, 3, 1, 2)

        coeffs_x1_out = self.resnet_gated(coeffs_x1_batch)
        coeffs_x2_temp = self.resnet_gated(coeffs_x2_batch)
        coeffs_x2_out = self.wvlet2_conv1(coeffs_x2_temp)

        coeffs_x = torch.cat([coeffs_x1_out, coeffs_x2_out], axis=1)
        out = self.wvletFusion(coeffs_x)
        out_size = nn.functional.interpolate(out, size=(8, 8), mode='bilinear', align_corners=True)
        out_size = out_size.view(out_size.size(0), out_size.size(1), out_size.size(2) * out_size.size(3))

        return out_size


    def wavelet_3d_transform(self, x, wavelet='haar', level=2):
        """
        对一个batch中每段视频的多帧图像序列进行3D小波变换
        :param image_sequence:
        :param wavelet:
        :param level:
        :return:
        """
        coeffs_3d_list = []
        self.level = level
        image_sequence = x.view(-1, 96, 112, 112)
        batch, num_frames, height, width = image_sequence.shape
        for index in range(batch):
            coeffs_3d = []
            # Step 1: 对每一帧进行二维小波分解
            frame_coeffs = []
            for i in range(num_frames):
                frame = image_sequence[index,i, :, :]
                frame = frame.cpu()
                coeffs = pywt.wavedec2(frame, wavelet, level=level)
                frame_coeffs.append(coeffs)

            # Step 2: 对帧间维度（时间轴）进行一维小波变换
            for coeff_level in range(level + 1):
                if coeff_level == 0:
                    # 处理低频分量（cA）
                    low_freq_sequence = np.stack([frame_coeffs[i][coeff_level] for i in range(num_frames)], axis=-1)
                    coeffs_t = pywt.wavedec(low_freq_sequence, wavelet, level=level, axis=-1)
                    coeffs_3d.append(('low', coeffs_t))
                else:
                    # 处理高频分量（cH, cV, cD）
                    for subband in range(3):  # 0: cH, 1: cV, 2: cD
                        high_freq_sequence = np.stack(
                            [frame_coeffs[i][coeff_level][subband] for i in range(num_frames)],
                            axis=-1)
                        coeffs_t = pywt.wavedec(high_freq_sequence, wavelet, level=level, axis=-1)
                        # high_freq_sequence_rec = pywt.waverec(coeffs_t, wavelet, axis=-1)
                        coeffs_3d.append(('high', subband, coeffs_t))
            coeffs_3d_list.append(coeffs_3d)


        return coeffs_3d_list