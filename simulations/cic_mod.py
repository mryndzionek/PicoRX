import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def tfmult(a, b):
    return np.flip(np.polymul(np.flip(a[0]), np.flip(b[0]))), np.flip(
        np.polymul(np.flip(a[1]), np.flip(b[1]))
    )


def find_nearest(x, a):
    return np.searchsorted(a, x, side="left")


FS = 480000

N = 4 # CIC order (number of integrator-comb sections)
R = 16  # downsampling factor (decimation rate)
P = 2  # downsampling factor due to next decimation stage(s)

worN = 512 * 4
wca = FS * ((1 / R) - 1 / (2 * R * P))
wca2 = wca + (FS // (P * R))
pband_end = FS // (2 * R)

cic1 = [1] + ([0] * (R - 1)) + [-1], [1, -1]
# w, h = sig.freqz(*cic1, worN=worN, fs=FS)
# plt.plot(w, 20 * np.log10(np.abs(h) / R), label=f"CIC{R}, order 1")

cicR = [1], [1]
for _ in range(N):
    cicR = tfmult(cicR, cic1)

w, h = sig.freqz(*cicR, worN=worN, fs=FS)

h_db = 20 * np.log10(np.abs(h) / (R**N))
idx = find_nearest(wca, w)
wca_db_before = h_db[idx]
plt.plot(w, h_db, label=f"CIC{R}, order {N}")

cicRm = [1], [1]
for i in range(N - 1):
    cicRm = tfmult(cic1, cicRm)

r = round(2 * R - ((R * R * P * 2) / (2 * R * P - R)))
d = 2 * R - r
print(f"Delay of the modified comb: {d}")
print(f"Polyphase delay: {R - r}")
cicRm = tfmult(cicRm, ([1] + ([0] * (d - 1)) + [-1], [1, -1]))

g = ((R ** (N - 1) * d))
w, hm = sig.freqz(*cicRm, worN=worN, fs=FS)
hm_db = 20 * np.log10(np.abs(hm) / g)
plt.plot(
    w,
    hm_db,
    label=f"CIC{R}, order {N}, modified",
)

plt.grid(True)
plt.xlim(0, FS // 2)
min_db, max_db = -180, 5
plt.ylim(min_db, max_db)

idx = find_nearest(wca2, w)
wca_db = hm_db[idx]

idx = find_nearest(pband_end, w)
passband_droop = h_db[idx] - hm_db[idx]
print(f"Passband droop: {round(passband_droop, 2)}dB")

plt.hlines(
    [h_db[idx], hm_db[idx]],
    0.8 * pband_end,
    1.2 * pband_end,
    linestyle="dashed",
    color="violet",
    label=f"Passband droop ({round(passband_droop, 2)}dB)",
)

plt.hlines([wca_db_before], 0.8 * wca, 1.2 * wca, linestyle="dashed", color="red")
plt.hlines([wca_db], 0.8 * wca2, 1.2 * wca2, linestyle="dashed", color="green")

plt.vlines(
    [wca],
    min_db,
    max_db,
    linestyle="dashed",
    color="red",
    label=f"Worst case attenuation before ({wca/1000}kHz, {round(wca_db_before, 2)}dB)",
)

plt.vlines(
    [wca2],
    min_db,
    max_db,
    linestyle="dashed",
    color="green",
    label=f"Worst case attenuation after ({wca2/1000}kHz, {round(wca_db, 2)}dB)",
)

plt.vlines(
    [pband_end],
    min_db,
    max_db,
    linestyle="dashed",
    color="black",
    label=f"Passband end ({FS // (1000 * 2 * R)}kHz)",
)

fb = (FS // 2) / (P * R)
for r in np.arange(0, (FS // 2) + FS / R, FS / R):
    plt.fill_between([r - fb, r + fb], min_db, max_db, alpha=0.2, color="gray")
plt.legend()
plt.xlabel("[Hz]")
plt.ylabel("[dB]")
plt.show()

corr = np.abs(hm) / g
corr[0] = corr[1]
w = w[0 : 1 + len(corr) // R]
corr = (1 / corr)[0 : 1 + len(corr) // R]
plt.plot(w, corr)
plt.title("CIC correction")
plt.grid(True)
plt.show()

print(f"CIC bit growth: {np.ceil(np.log2(g))}")
print("CIC corrections:")
print(",".join([str(int(round(256 * i))) for i in corr]))
