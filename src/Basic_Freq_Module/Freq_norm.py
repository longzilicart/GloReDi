# normalization in frequency domain by longzili
import numpy as np
import torch




def dct_normalize_torch(dct_results, method='min-max'):
    if method == 'min-max':
        min_val = torch.min(dct_results)
        max_val = torch.max(dct_results)
        normalized = (dct_results - min_val) / (max_val - min_val)

    elif method == 'z-score':
        mean = torch.mean(dct_results)
        std = torch.std(dct_results)
        normalized = (dct_results - mean) / std

    elif method == 'magnitude':
        magnitude = dct_results.abs()
        phase = dct_results.angle()
        mag_min = torch.min(magnitude)
        mag_max = torch.max(magnitude)
        normalized_magnitude = (magnitude - mag_min) / (mag_max - mag_min)
        normalized = normalized_magnitude * torch.exp(1j * phase)

    elif method == 'log-scaling':
        log_dct = torch.log1p(dct_results.abs())
        min_val = torch.min(log_dct)
        max_val = torch.max(log_dct)
        normalized = (log_dct - min_val) / (max_val - min_val)
        normalized = normalized * torch.sign(dct_results)

    elif method == 'sigmoid':
        normalized = 1 / (1 + torch.exp(-dct_results))

    else:
        raise ValueError(f"not support method: {method}")

    return normalized





def fft_normalize(fft_results, method='magnitude'):
    '''
    '''
    if method == 'magnitude':
        magnitude = fft_results.abs()
        normalized = (magnitude - torch.min(magnitude)) / (torch.max(magnitude) - torch.min(magnitude))

    elif method == 'z-score':
        magnitude = fft_results.abs()
        mean = torch.mean(magnitude)
        std = torch.std(magnitude)
        normalized = (magnitude - mean) / std

    elif method == 'log-scaling':
        magnitude = fft_results.abs()
        log_magnitude = torch.log1p(magnitude)
        normalized = (log_magnitude - torch.min(log_magnitude)) / (torch.max(log_magnitude) - torch.min(log_magnitude))

    elif method == 'phase':
        phase = fft_results.angle() / torch.pi
        normalized = (phase + 1) / 2

    elif method == 'sigmoid':
        magnitude = fft_results.abs()
        normalized = 1 / (1 + torch.exp(-magnitude))

    else:
        raise ValueError(f"not support method: {method}")

    return normalized






if __name__ == '__main__':
    pass