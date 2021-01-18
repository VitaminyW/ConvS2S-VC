import librosa
import numpy as np
import os
import pyworld
import pysptk
import time

log_pias = 0.0000000000000000000001  # 防止f0=0使得log值取负无穷


def PrepareDate(path, logf0s_mean, logf0s_std, mcs_mean, mcs_std,
                sampling_rate=16000, num_mcep=29, frame_period=5.0):
    """
    获取模型训练的数据
    :param path: wav文件地址
    :param logf0s_mean: 语者logf0的均值
    :param logf0s_std: 语者logf0的方差
    :param mcs_mean: 语者mcs的均值
    :param mcs_std: 语者mcs的方差
    :param sampling_rate: 取样频率
    :param num_mcep: mccs的数量
    :param frame_period: 时间窗
    :return: 93*t 维度的数据矩阵
    """
    wav, _ = librosa.load(path, sr=sampling_rate, mono=True)
    wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
    f0s, timeaxes, sps, aps, mcs = world_encode_data(wavs=[wav],
                                                     fs=sampling_rate,
                                                     frame_period=frame_period,
                                                     num_mcep=num_mcep)
    f0s += log_pias
    f0s = f0s[0].reshape((1, f0s[0].shape[0]))
    f0s = np.log(f0s)
    mcs_mean = mcs_mean.reshape((mcs_mean.shape[0], 1))
    mcs_std = mcs_std.reshape((mcs_std.shape[0], 1))
    mcs = mcs[0].T
    f0s = (f0s - logf0s_mean) / logf0s_std
    mcs = (mcs - mcs_mean) / mcs_std

    num_samples = int(1000 * len(wav) / sampling_rate / 5 + 1)
    gap = int(len(wav) / num_samples)
    begin = 0
    end = gap
    v_uv = []
    for i in range(num_samples - 1):
        v_uv.append(judge(wav[begin:end]))
        begin += gap
        end += gap
    v_uv.append(judge(wav[end:]))
    v_uv = np.array(v_uv).reshape((1, len(v_uv)))
    result = np.concatenate([f0s, mcs, v_uv], axis=0)
    s = int(result.shape[1] / 3)
    temp = []
    for i in range(3):
        temp.append(result[:, s * i:s * (i + 1)])
    result = np.concatenate(temp, axis=0)
    return result


def getDataForPrepare(data_dir, batch, sampling_rate=16000, num_mcep=29, frame_period=5.0):
    """
    获取读者所有音频的均值和方差
    :param data_dir: 文件目录
    :param batch: 一次加载进内存的音频数量
    :param sampling_rate: 取样频率
    :param num_mcep: 要获取的mcc数量
    :param frame_period: 时间窗大小
    :return: logf0s_mean, logf0s_std, mcs_mean, mcs_std
    """
    wavs = load_wavs(wav_dir=data_dir, sr=sampling_rate)
    num = int(len(wavs) / batch)
    f0_num = 0
    mcs_num = 0
    logf0s_mean = 0
    mcs_mean = None
    for i in range(num):
        f0s, timeaxes, sps, aps, mcs = world_encode_data(
            wavs=wavs[batch * i:batch * (i + 1)] if i < num - 1 else wavs[batch * i:],
            fs=sampling_rate,
            frame_period=frame_period,
            num_mcep=num_mcep)
        print(f0s)
        total_logf0 = 0
        for f0 in f0s:
            f0_num += f0.shape[0]
            f0 += log_pias
            total_logf0 += np.log(f0).sum()
        logf0s_mean += total_logf0
        total_mc = None
        for mc in mcs:
            if total_mc is None:
                total_mc = mc.sum(axis=0)
            else:
                total_mc += mc.sum(axis=0)
            mcs_num += mc.shape[0]
        if mcs_mean is None:
            mcs_mean = total_mc
        else:
            mcs_mean += total_mc
    logf0s_mean /= f0_num
    mcs_mean /= mcs_num
    logf0s_std = 0
    mcs_std = None
    for i in range(num):
        f0s, timeaxes, sps, aps, mcs = world_encode_data(
            wavs=wavs[batch * i:batch * (i + 1)] if i < num - 1 else wavs[batch * i:],
            fs=sampling_rate,
            frame_period=frame_period,
            num_mcep=num_mcep)
        total_logf0 = 0
        for f0 in f0s:
            f0 += log_pias
            total_logf0 += ((np.log(f0) - logf0s_mean) ** 2).sum()
        logf0s_std += total_logf0
        total_mc = None
        for mc in mcs:
            if total_mc is None:
                total_mc = ((mc - mcs_mean) ** 2).sum(axis=0)
            else:
                total_mc += ((mc - mcs_mean) ** 2).sum(axis=0)
        if mcs_std is None:
            mcs_std = total_mc
        else:
            mcs_std += total_mc
    logf0s_std /= f0_num
    mcs_std /= f0_num
    logf0s_std = logf0s_std ** 0.5
    mcs_std = mcs_std ** 0.5
    return logf0s_mean, logf0s_std, mcs_mean, mcs_std


def load_wavs(wav_dir, sr):
    """
    :param wav_dir:储存wav文件的地址
    :param sr: 音频取样频率
    :return: 音频文件
    """

    debug_num = 0
    wavs = list()
    for file in os.listdir(wav_dir):
        if (file.count("wav") == 0):
            continue
        """
        debug_num += 1
        if (debug_num > 100):
            break
        """
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr, mono=True)
        wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
        # wav = wav.astype(np.float64)
        wavs.append(wav)

    return wavs


def world_decompose(wav, fs, frame_period=5.0, num_mcep=36):
    """
    获取音频特征
    :param wav: 音频np数组
    :param fs: 取样频率
    :param frame_period: 窗口大小
    :param num_mcep: 获取的mccs数目
    :return:
    """
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.sp2mc(sp, order=num_mcep - 1, alpha=alpha)

    return f0, timeaxis, sp, ap, mc


def world_encode_spectral_envelop(sp, fs, dim=24):
    """
    获取包络频谱
    """
    # Get Mel-cepstral coefficients (MCEPs)

    # sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp


def world_decode_mc(mc, fs):
    """
    解码mccs
    """
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    # coded_sp = coded_sp.astype(np.float32)
    # coded_sp = np.ascontiguousarray(coded_sp)
    alpha = pysptk.util.mcepalpha(fs)
    sp = pysptk.mc2sp(mc, alpha, fftlen)
    # decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return sp


def world_encode_data(wavs, fs, frame_period=5.0, num_mcep=29):
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    mcs = list()

    for wav in wavs:
        f0, timeaxis, sp, ap, mc = world_decompose(wav=wav, fs=fs, frame_period=frame_period, num_mcep=num_mcep)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        mcs.append(mc)

    return f0s, timeaxes, sps, aps, mcs


def transpose_in_list(lst):
    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


def world_decode_data(coded_sps, fs):
    decoded_sps = list()

    for coded_sp in coded_sps:
        decoded_sp = world_encode_spectral_envelop(coded_sp, fs)
        decoded_sps.append(decoded_sp)

    return decoded_sps


def world_speech_synthesis(f0, sp, ap, fs, frame_period):
    # decoded_sp = decoded_sp.astype(np.float64)
    wav = pyworld.synthesize(f0, sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)

    return wav


def world_synthesis_data(f0s, decoded_sps, aps, fs, frame_period):
    wavs = list()

    for f0, decoded_sp, ap in zip(f0s, decoded_sps, aps):
        wav = world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period)
        wavs.append(wav)

    return wavs


def mcs_normalization_fit_transoform(mcs):
    mcs_concatenated = np.concatenate(mcs, axis=1)
    mcs_mean = np.mean(mcs_concatenated, axis=1, keepdims=True)
    mcs_std = np.std(mcs_concatenated, axis=1, keepdims=True)

    mcs_normalized = list()
    for mc in mcs:
        mcs_normalized.append((mc - mcs_mean) / mcs_std)

    return mcs_normalized, mcs_mean, mcs_std


def coded_sps_normalization_transoform(coded_sps, coded_sps_mean, coded_sps_std):
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized


def coded_sps_normalization_inverse_transoform(normalized_coded_sps, coded_sps_mean, coded_sps_std):
    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps


def coded_sp_padding(coded_sp, multiple=4):
    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values=0)

    return coded_sp_padded


def wav_padding(wav, sr, frame_period, multiple=4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int(
        (np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (
                sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values=0)

    nlen = len(wav_padded) + 80
    a = 2 ** 5
    wav_padded = np.pad(wav_padded, (0, (a - (nlen // 80) % a) * 80 - (nlen % 80)), 'constant', constant_values=0)

    return wav_padded


def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std


def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):
    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted


def wavs_to_specs(wavs, n_fft=1024, hop_length=None):
    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        stfts.append(stft)

    return stfts


def wavs_to_mfccs(wavs, sr, n_fft=1024, hop_length=None, n_mels=128, n_mfcc=24):
    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
        mfccs.append(mfcc)

    return mfccs


def sample_train_data(dataset_A, dataset_B, n_frames=128):
    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:, start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:, start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B


def judge(target):
    """
    获取清浊音
    :param target: 音频
    :return:
    """
    array1 = np.array(target)
    length = len(array1)
    array2 = array1[1:length]
    array2 = np.append(array2, 0)
    sign = array1 * array2
    sign2 = np.where(sign < 0)
    rate = len(sign2[0]) / (length - 1)  # 过零率的计算
    # 短时对数能量
    array3 = array1.astype(np.float)
    E = 10 * np.log10(np.sum(array3 ** 2) / length)
    if E > -110:
        if rate > 0.48:
            flag = 1  # 清音
        else:
            flag = 0  # 浊音
    else:
        return -1
    return flag
