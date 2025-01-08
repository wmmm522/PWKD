import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt



def dwt_transform(x, wavelet='db1', levels=3):
    coeffs = pywt.wavedec2(x.squeeze().cpu().numpy(), wavelet, level=levels)
    return coeffs



def idwt_transform(coeffs, wavelet='db1'):
    reconstructed = pywt.waverec2(coeffs, wavelet)
    return torch.tensor(reconstructed).unsqueeze(0).unsqueeze(0)


class MVRM(nn.Module):
    def __init__(self, levels=3):
        super(MVRM, self).__init__()
        self.levels = levels
        self.conv_HH1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv_HH2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        self.conv_HH3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, padding=3)

    def forward(self, x):

        coeffs = dwt_transform(x, levels=self.levels)


        LL, (LH1, HL1, HH1), (LH2, HL2, HH2), (LH3, HL3, HH3) = coeffs


        HH1_conv = self.conv_HH1(torch.tensor(HH1).unsqueeze(0).to(x.device))
        HH2_conv = self.conv_HH2(torch.tensor(HH2).unsqueeze(0).to(x.device))
        HH3_conv = self.conv_HH3(torch.tensor(HH3).unsqueeze(0).to(x.device))

        coeffs_mvrmed = [LL, (LH1, HL1, HH1_conv.squeeze().cpu().numpy()),
                         (LH2, HL2, HH2_conv.squeeze().cpu().numpy()),
                         (LH3, HL3, HH3_conv.squeeze().cpu().numpy())]

        y = idwt_transform(coeffs_mvrmed)
        return y.to(x.device)


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.mvrms = nn.ModuleList([MVRM(levels=3) for _ in range(num_intermediate_layers)])  # 假设有多个中间层

    def forward(self, teacher_features, student_features):
        distill_loss = 0.0
        for i, (teacher_feature, student_feature) in enumerate(zip(teacher_features, student_features)):

            student_feature_mvrmed = self.mvrms[i](student_feature)


            soft_teacher_output = F.log_softmax(teacher_feature / self.temperature, dim=1)
            soft_student_output = F.softmax(student_feature_mvrmed / self.temperature, dim=1)

  
            distill_loss += F.kl_div(soft_student_output, soft_teacher_output, reduction='batchmean') * (
                        self.temperature ** 2)

        return distill_loss