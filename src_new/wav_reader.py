import librosa
import numpy as np
from scipy.signal import lfilter, butter

import sigproc # see details: https://www.cnblogs.com/zhuimengzhe/p/10223510.html
import constants as c
import os


def load_wav(filename, sample_rate):
	audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
	audio = audio.flatten()# 按行方向降为 1 维
	return audio


def normalize_frames(m,epsilon=1e-12):
	return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def remove_dc_and_dither(sin, sample_rate):
	if sample_rate == 16e3:
		alpha = 0.99
	elif sample_rate == 8e3:
		alpha = 0.999
	else:
		print("Sample rate must be 16kHz or 8kHz only")
		exit(1)
	sin = lfilter([1,-1], [1,-alpha], sin)
	dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
	spow = np.std(dither)
	sout = sin + 1e-6 * spow * dither
	return sout


def get_fft_spectrum(filename, buckets, mode="train"):
	signal = load_wav(filename.split("\n")[0],c.SAMPLE_RATE)
	signal *= 2**15

	# 得到 短时傅里叶变化 频谱图
	# get FFT spectrum
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE) # 数字滤波器,去除直流和颤动成分
	signal = sigproc.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA) # 对输入信号进行预加重
	frames = sigproc.framesig(signal, frame_len=c.FRAME_LEN*c.SAMPLE_RATE, frame_step=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming) # 将信号框成重叠帧
	# print(frames)
	# print(frames.shape) # 帧的个数 x 每一帧包含的采样点数 (None x 400)
	# os.system("pause")
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT)) # 只取幅度
	# n : 输出的转换轴的长度。如果n小于输入的长度，则裁剪输入。如果较大，则输入将填充零。如果未指定n，则使用输入沿由axis指定的轴的长度。
	fft_norm = normalize_frames(fft.T) # 减去均值，除以标准差  (512 x None)

	# for kk in buckets:
	# 	print("**",kk)
	# print("max",max(kkk for kkk in buckets))
	# os.system("pause")

	# truncate to max bucket sizes
	rsize = max(k for k in buckets if k <= fft_norm.shape[1]) # k<=1501 取本语音能通过卷积层的最大长度,转化为整百
	rstart = int((fft_norm.shape[1]-rsize)/2)
	l = len(fft_norm[0]) - len(fft_norm[0]) % 100

	out = fft_norm[:,rstart:rstart+300] if mode == "train" else fft_norm[:,:l]
	# print(out.shape)
	# os.system("pause")

	# out = fft_norm[:, rstart:rstart + rsize]

	return out
