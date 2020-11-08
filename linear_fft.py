import time
import numpy as np
from queue import deque
from queue import Queue
from scipy.fftpack import fft, fftshift

global cycle
cycle = 0


def DFT(x):
    """Compute the Discrete Fourier Transform of the 1D array x"""
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    N = len(x)
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 2:  # this cutoff should be optimized
        return DFT(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2.0j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])


def complex_mult(x, y):
    global cycle
    list_3 = []
    for i in range(len(x)):
        real_part = x[i].real * y[i].real - x[i].imag * y[i].imag
        image_part = x[i].imag * y[i].real + x[i].real * y[i].imag
        complex_result = real_part + image_part * 1j
        list_3.append(complex_result)
        cycle += 1

    return list_3


def rFFT(x):
    """
    Recursive FFT implementation.
    References
      -- http://www.cse.uiuc.edu/iem/fft/rcrsvfft/
      -- "A Simple and Efficient FFT Implementation in C++"
          by Vlodymyr Myrnyy
    """

    n = len(x)
    if n == 1:
        return x

    w = getTwiddle(n)
    m = n // 2
    X = np.ones(m, float) * 1j
    Y = np.ones(m, float) * 1j

    for k in range(m):
        X[k] = x[2 * k]
        Y[k] = x[2 * k + 1]

    X = rFFT(X)
    Y = rFFT(Y)
    F = np.ones(n, float) * 1j

    for k in range(n):
        i = (k % m)
        F[k] = X[i] + w[k] * Y[i]
        global cycle
        cycle += 1

    return F


def getTwiddle(NFFT):
    """Generate the twiddle factors"""

    W = np.r_[[1.0 + 1.0j] * NFFT]
    for k in range(NFFT):
        W[k] = np.exp(-2.0j * np.pi * k / NFFT)

    return W


def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    N = len(x)

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 2)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, np.array(x).reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()


class LinearArrayCell:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.cell_index = 0
        self.cell_input = None
        self.single_in = 0
        self.single_out = 0
        self.data_to_compute_1 = Queue(maxsize=self.cell_size)
        self.data_to_compute_2 = Queue(maxsize=self.cell_size)
        self.cell_shift = Queue()
        self.cell_partial_result = Queue()
        self.cell_output = Queue()
        self.signal_index = 0

    def connect(self, cell_index, array, array_size, iterations):
        self.cell_index = cell_index
        if iterations % array_size == 0:  # a group of data completed loop in all cells
            if self.signal_index > 0:
                self.clear_shift()
            self.cell_input = array.input[self.signal_index][self.cell_index]
            self.signal_index += 1
        else:
            self.cell_input = array.cells[self.cell_index - 1]  # shifting registers

    def cell_read(self):  # load all data needed for a cell
        global cycle
        if type(self.cell_input) is Queue:  # from input FIFO
            for _ in range(self.cell_size):
                if self.cell_input.empty():
                    self.single_in = 0
                else:
                    self.single_in = self.cell_input.get()
                self.data_to_compute_1.put(self.single_in)
                self.data_to_compute_2.put(self.single_in.real - self.single_in.imag * 1j)  # conjugate
                cycle += 1
        else:  # from shift registers (only for data, not for conjugate(data))
            for _ in range(self.cell_size):
                self.single_in = self.cell_input.cell_shift.get()
                self.data_to_compute_1.put(self.single_in)
                cycle += 1
                # self.data_to_compute_2.put(self.single_in.real - self.single_in.imag * 1j)

    def compute(self, iterations):
        list_1 = list(self.data_to_compute_1.queue)
        list_2 = list(self.data_to_compute_2.queue)
        list_3 = []
        list_3 = complex_mult(list_1, list_2)
        '''
        for i in range(len(list_1)):
            real_part = list_1[i].real * list_2[i].real - list_1[i].imag * list_2[i].imag
            image_part = list_1[i].imag * list_2[i].real + list_1[i].real * list_2[i].imag
            complex_result = real_part + image_part * 1j
            list_3.append(complex_result)
            global cycle
            cycle += 1
        '''
        # fft_result = DFT(list_3)
        # fft_result = FFT(list_3)
        fft_result = rFFT(list_3)
        # fft_result = FFT_vectorized(list_3)
        # fft_result = fft(list_3)
        # print(f'Compare DFT with built-in FFT at PE {iterations}:', np.allclose(DFT(list_3), fft(list_3)))
        # print(f'Compare FFT with built-in FFT at PE {iterations}:', np.allclose(FFT(list_3), fft(list_3)))
        # print(f'Compare rFFT with built-in FFT at PE {iterations}:', np.allclose(rFFT(list_3), fft(list_3)))
        # print(f'Compare FFT with built-in FFT at PE {iterations}:', np.allclose(FFT_vectorized(list_3), fft(list_3)))
        fft_shift_results = fftshift(fft_result)[registers // 2 - 8: registers // 2 + 8]  # take middle 16-bit
        self.cell_output.queue = deque(fft_shift_results)
        '''
        ### To be continued (Alpha profile) ### 
        for j in range(len(fft_shift_results)):
            alpha_partial = np.absolute(fft_shift_results[j])
            alpha_final = np.absolute(self.cell_partial_result)
            if alpha_partial > alpha_final:
                self.cell_partial_result = alpha_partial
        '''
        for _ in range(self.cell_size):
            self.single_out = self.data_to_compute_1.get()
            self.cell_shift.put(self.single_out)

        # if iterations == total_iter:
        # self.cell_output.queue = deque(self.cell_partial_result)

    def clear_shift(self):  # to clear shift data from cell_shift queue when complete an input signal
        for _ in range(self.cell_size):
            self.cell_shift.get()
            self.data_to_compute_2.get()


class LinearArray:
    def __init__(self, array_size, cell_size, fifo_input):
        self.array_size = array_size
        self.cell_size = cell_size
        self.input = fifo_input
        self.iterations = 0
        self.cells = []
        self.result = [[Queue() for _ in range(self.array_size)] for _ in range(len(self.input))]

        for _ in range(self.array_size):
            cell = LinearArrayCell(self.cell_size)
            self.cells.append(cell)

    def connect(self):
        for cell_index, cell in enumerate(self.cells):
            cell.connect(cell_index, self, self.array_size, self.iterations)

    def read(self):
        for cell in self.cells:
            cell.cell_read()

    def compute(self):
        for cell in self.cells:
            cell.compute(self.iterations)
            for _ in range(cell.cell_size // 2):
                self.result[cell.signal_index - 1][cell.cell_index].put(cell.cell_output.get())

    def run(self, total_iterations):
        for _ in range(total_iterations):
            self.connect()
            self.read()
            self.compute()
            self.iterations += 1
        return self.result


signals = 1
pes = 1  # 256
registers = 32
total_iter = signals * pes
input_queue = [[Queue() for _ in range(pes)] for _ in range(signals)]
read_cycle = registers * pes
compute_cycle = registers + registers * np.log2(registers)
shift_cycle = registers
total_cycle = int(read_cycle + compute_cycle + (shift_cycle + compute_cycle) * (pes - 1))
total_time = total_cycle * 2 / 1000
print('Theoretically, total number of cycles = {:d}, Time on FPGA = {:f} us at 500MHz.'.format(total_cycle, total_time))
# print(f'No. of cycles = {int(total_cycle)}, Execution time = {total_cycle * 2 / 1000} us at 500MHz.')

for signal in range(signals):
    for pe in range(pes):
        if signal == 0:
            for index in range(pe * registers, (pe + 1) * registers):
                complex_data = index / 128 + (index + 1) / 128 * 1j
                input_queue[signal][pe].put(complex_data)
        if signal > 0:
            for index in reversed(range(pe * registers, (pe + 1) * registers)):
                complex_data = index / 128 + (index + 1) / 128 * 1j
                input_queue[signal][pe].put(complex_data)

# print(list(input_queue[0][-1].queue))

myArray = LinearArray(pes, registers, input_queue)
start_time = time.time()
res = myArray.run(total_iter)  # run (signal*pes) times
end_time = time.time()
cpu_time = end_time - start_time
pe_cycle = cycle // pes
print('---{:6.2f} seconds on CPU---'.format(cpu_time))
print('Real total number of cycles on PE = {}'.format(pe_cycle))

'''
for i in range(signals):
    for j in range(pes):
        # print('PE%d:' %j, list(res[i][j].queue))
        # print('PE{:d} output: {}'.format(j, ['%.5f, %.5f' % (element.real,
        # element.imag) for element in list(res[i][j].queue)]))
        print(len(list(res[i][j].queue)))
'''
