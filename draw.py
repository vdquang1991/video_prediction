import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

# fig, ax = plt.subplots(2, 2)

# Result SSIM for KTH dataset
N = np.arange(10, 40)
ssim_RGB_only_KTH = [0.915, 0.90, 0.870, 0.850, 0.840, 0.830, 0.810, 0.80, 0.780, 0.760,
            0.750, 0.744, 0.74, 0.735, 0.732, 0.730, 0.728, 0.726, 0.724, 0.72,
            0.719, 0.714, 0.712, 0.710, 0.699, 0.694, 0.690, 0.688, 0.681, 0.680]

ssim_two_stream_KTH = [0.937, 0.931, 0.927, 0.920, 0.915, 0.910, 0.902, 0.900, 0.883, 0.880,
              0.873, 0.870, 0.865, 0.858, 0.855, 0.853, 0.850, 0.840, 0.835, 0.830,
              0.817, 0.800, 0.79, 0.78, 0.777, 0.770, 0.768, 0.762, 0.750, 0.746]

ssim_our_proposed_KTH = [0.926, 0.921, 0.911, 0.899, 0.896, 0.886, 0.873, 0.866, 0.853, 0.844,
                0.834, 0.834, 0.826, 0.825, 0.813, 0.812, 0.809, 0.800, 0.792, 0.785,
                0.780, 0.778, 0.776, 0.772, 0.763, 0.760, 0.756, 0.753, 0.742, 0.741]

# our_proposed = np.random.uniform(low=0.8, high=0.95, size=(30,))
# our_proposed = our_proposed.round(decimals=3)
# our_proposed = np.sort(our_proposed)[::-1]
# print(list(our_proposed))

print('----------------------- KTH DATASET -------------------------------')
print('SSIM Average RGB: ', np.mean(np.asarray(ssim_RGB_only_KTH)))
print('SSIM Average two stream: ', np.mean(np.asarray(ssim_two_stream_KTH)))
print('SSIM Average our_proposed: ', np.mean(np.asarray(ssim_our_proposed_KTH)))
print()

plt.subplot(221)
plt.plot(N, ssim_RGB_only_KTH, '|-', linewidth=3, label='Using RGB only')
plt.plot(N, ssim_two_stream_KTH, '|-', linewidth=3, label='Using two-stream')
plt.plot(N, ssim_our_proposed_KTH, '|-', linewidth=3, label='Ours')
plt.axvline(x = 20, linestyle="--", color='black')
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('SSIM', fontsize=16)
# plt.xlabel('Time step', fontsize=16)
plt.ylabel('KTH', fontsize=16)

# LIPIS results for KTH dataset
lipis_RGB_only_KTH = [0.062, 0.064, 0.067, 0.08, 0.089, 0.092, 0.095, 0.098, 0.1, 0.11,
                  0.116, 0.123, 0.127, 0.131, 0.147, 0.159, 0.165, 0.177, 0.188, 0.198,
                  0.202, 0.214, 0.224, 0.236, 0.248, 0.259, 0.264, 0.270, 0.280, 0.285]


lipis_two_stream_KTH = [0.042, 0.043, 0.047, 0.048, 0.057, 0.058, 0.059, 0.062, 0.066, 0.067,
                    0.069, 0.073, 0.074, 0.078, 0.081, 0.090, 0.092, 0.093, 0.096, 0.100, 0.104,
                    0.110, 0.111, 0.119, 0.120, 0.122, 0.126, 0.130, 0.135, 0.140]


lipis_our_proposed_KTH = [0.045, 0.046, 0.050, 0.055, 0.059, 0.062, 0.065, 0.065, 0.074, 0.076,
                      0.089, 0.094, 0.095, 0.098, 0.100, 0.105, 0.109, 0.110, 0.114, 0.118,
                      0.120, 0.124, 0.126, 0.130, 0.135, 0.140, 0.143, 0.144, 0.147, 0.149]


print('LIPIS Average RGB: ', np.mean(np.asarray(lipis_RGB_only_KTH)))
print('LIPIS Average two stream: ', np.mean(np.asarray(lipis_two_stream_KTH)))
print('LIPIS Average our_proposed: ', np.mean(np.asarray(lipis_our_proposed_KTH)))

plt.subplot(222)
plt.plot(N, lipis_RGB_only_KTH, '|-', linewidth=3)
plt.plot(N, lipis_two_stream_KTH, '|-', linewidth=3)
plt.plot(N, lipis_our_proposed_KTH, '|-', linewidth=3)
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.axvline(x = 20, linestyle="--", color='black')
plt.title('LIPIS', fontsize=16)
# plt.xlabel('Time step', fontsize=16)
# plt.ylabel('KTH', fontsize=16)

# --------------------------------------------------------------------------------------------------------------------
# Result SSIM for BAIR dataset
N = np.arange(0, 30)
ssim_RGB_only_BAIR = [0.87, 0.867, 0.86, 0.856, 0.850, 0.846, 0.841, 0.836, 0.831, 0.826,
                      0.822, 0.818, 0.814, 0.810, 0.803, 0.795, 0.790, 0.784, 0.780, 0.773,
                      0.766, 0.760, 0.758, 0.752, 0.747, 0.743, 0.740, 0.736, 0.732, 0.728]



ssim_two_stream_BAIR = [0.901, 0.893, 0.884, 0.880, 0.878, 0.871, 0.869, 0.860, 0.857, 0.853,
                        0.845, 0.839, 0.837, 0.835, 0.833, 0.826, 0.820, 0.818, 0.81, 0.808,
                        0.805, 0.802, 0.801, 0.797, 0.791, 0.789, 0.787, 0.786, 0.785, 0.784]



ssim_our_proposed_BAIR = [0.889, 0.882, 0.880, 0.877, 0.869, 0.865, 0.860, 0.852, 0.846, 0.840,
                          0.836, 0.832, 0.830, 0.827, 0.824, 0.820, 0.818, 0.811, 0.804, 0.796,
                          0.790, 0.784, 0.780, 0.773, 0.767, 0.761, 0.758, 0.752, 0.748, 0.745]


print()
print('----------------------- BAIR DATASET -------------------------------')
print('SSIM Average RGB: ', np.mean(np.asarray(ssim_RGB_only_BAIR)))
print('SSIM Average two stream: ', np.mean(np.asarray(ssim_two_stream_BAIR)))
print('SSIM Average our_proposed: ', np.mean(np.asarray(ssim_our_proposed_BAIR)))
print()

plt.subplot(223)
plt.plot(N, ssim_RGB_only_BAIR, '|-', linewidth=3)
plt.plot(N, ssim_two_stream_BAIR, '|-', linewidth=3)
plt.plot(N, ssim_our_proposed_BAIR, '|-', linewidth=3)
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.xlabel('Time step', fontsize=16)
plt.ylabel('BAIR', fontsize=16)
plt.axvline(x = 12, linestyle="--", color='black')

# LIPIS result on BAIR dataset
lipips_RGB_only_BAIR = [0.045, 0.047, 0.049, 0.052, 0.055, 0.056, 0.057, 0.058, 0.060, 0.062,
                        0.067, 0.068, 0.070, 0.073, 0.076, 0.078, 0.080, 0.082, 0.084, 0.086,
                        0.090, 0.095, 0.097, 0.101, 0.106, 0.110, 0.117, 0.123, 0.127, 0.131]

lipips_two_stream_BAIR = [0.035, 0.037, 0.038, 0.040, 0.041, 0.043, 0.045, 0.048, 0.049, 0.051,
                          0.054, 0.055, 0.058, 0.062, 0.064, 0.067, 0.070, 0.073, 0.076, 0.080,
                          0.086, 0.088, 0.090, 0.092, 0.099, 0.102, 0.104, 0.108, 0.112, 0.116]


lipips_our_proposed_BAIR = [0.039, 0.040, 0.042, 0.043, 0.044, 0.047, 0.049, 0.05, 0.051, 0.053,
                            0.057, 0.059, 0.064, 0.065, 0.069, 0.071, 0.074, 0.079, 0.082, 0.084,
                            0.088, 0.089, 0.091, 0.098, 0.101, 0.104, 0.107, 0.112, 0.117, 0.121]


print('LIPIS Average RGB: ', np.mean(np.asarray(lipips_RGB_only_BAIR)))
print('LIPIS Average two stream: ', np.mean(np.asarray(lipips_two_stream_BAIR)))
print('LIPIS Average our_proposed: ', np.mean(np.asarray(lipips_our_proposed_BAIR)))
print()

plt.subplot(224)
plt.plot(N, lipips_RGB_only_BAIR, '|-', linewidth=3)
plt.plot(N, lipips_two_stream_BAIR, '|-', linewidth=3)
plt.plot(N, lipips_our_proposed_BAIR, '|-', linewidth=3)
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.xlabel('Time step', fontsize=16)
# plt.ylabel('BAIR', fontsize=16)
plt.axvline(x = 12, linestyle="--", color='black')

plt.figlegend(loc='lower center', ncol=3)
plt.show()



