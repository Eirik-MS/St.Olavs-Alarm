import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures

# Define the file path
file_path = './LTSpice/LTC3204-3.3_1.txt'

# Define the number of rows to skip (if necessary)
skip_rows = 1

# Define the number of rows to read (if necessary)
num_rows = None

# Read the data from the file
data = pd.read_csv(file_path, sep='\t', skiprows=skip_rows, nrows=num_rows).values
# Extract the columns from the data
time = data[:, 0]
V_in = data[:, 1]
V_out = data[:, 2]
I_1 = data[:, 3]
I_Bat = data[:, 4]

sampling_rate = 1 / (time[1] - time[0])


def frequency_domain_analysis(data, sampling_rate):
    ptp = np.ptp(data[1000:])
    n = len(data)
    
    # Perform FFT on the data
    fft_data = np.fft.fft(data)

    # Calculate the frequency values
    freq = np.fft.fftfreq(len(data), d=1/sampling_rate)
    
    # Plot only positive frequencies
    positive_frequencies = freq[freq > 0]
    positive_fft_values = 2.0/n * np.abs(fft_data[:n//2])*1000
    
    positive_frequencies = positive_frequencies[35:]
    positive_fft_values = positive_fft_values[35:]
    
    # Find the index of the maximum amplitude (excluding the DC component)
    max_amp_index = np.argmax(np.abs(positive_fft_values[1:])) + 1
    
    # Calculate the frequency corresponding to the maximum amplitude
    noise_frequency = positive_frequencies[max_amp_index]

    # Calculate the amplitude of the noise
    noise_amplitude = np.abs(positive_fft_values[max_amp_index])
    
    # Plot the frequency domain data on a logarithmic scale
    #plt.semilogx(positive_frequencies, positive_fft_values, label='Real Part')
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Amplitude')
    #plt.title('Frequency Domain Data')
    #plt.show()

    return [ptp, noise_frequency, noise_amplitude, positive_frequencies, positive_fft_values]


with concurrent.futures.ThreadPoolExecutor() as executor:
    V_in_analysis = executor.submit(frequency_domain_analysis, V_in, sampling_rate)
    V_out_analysis = executor.submit(frequency_domain_analysis, V_out, sampling_rate)

    V_in_analysis = V_in_analysis.result()
    V_out_analysis = V_out_analysis.result()

analysis = {'V_in': V_in_analysis, 'V_out': V_out_analysis}

for key, value in analysis.items():
    print(f"{key:^25}")
    print(f"{key} Peak to Peak Voltage: {value[0]:.5f}")
    print(f"{key} Noise Frequency:      {value[1]:.5f}")
    print(f"{key} Noise Amplitude:      {value[2]:.5f}")
    # Plot the frequency domain data on a logarithmic scale
    if (len(value[3])!= len(value[4])):
        print(f"Length of {key} frequency domain data is not equal")
        if (len(value[3])>= len(value[4])):
            value[3] = value[3][:len(value[4])]
        else:
            value[4] = value[4][:len(value[3])]
    plt.semilogx(value[3], value[4], label='Real Part')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'{key} Frequency Domain Data')
    plt.savefig(f"{file_path[10:-4]}_{key}_frequency_domain.png", dpi=1200)
    plt.clf()
    


# Define the function to average the data
def average_data(data, n):
    return np.mean(data[:len(data)//n*n].reshape(-1, n), 1)

# Define the number of samples to combine
n = 10

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Apply the average_data function to each data array
    V_in_avg = executor.submit(average_data, V_in, n)
    V_out_avg = executor.submit(average_data, V_out, n)
    I_1_avg = executor.submit(average_data, I_1, n)
    I_Bat_avg = executor.submit(average_data, I_Bat, n)
    time_avg = executor.submit(average_data, time, n)

    # Get the results from the futures
    V_in_avg = V_in_avg.result()
    V_out_avg = V_out_avg.result()
    I_1_avg = I_1_avg.result()
    I_Bat_avg = I_Bat_avg.result()
    time_avg = time_avg.result()


V_in_avg = V_in_avg[1000:]
V_out_avg = V_out_avg[1000:]
I_1_avg = I_1_avg[1000:]
I_Bat_avg = I_Bat_avg[1000:]
time_avg = time_avg[1000:]


# Plot V_in and V_out together
plt.subplot(2, 1, 1)
plt.plot(time_avg, V_in_avg, label='V_in')
plt.plot(time_avg, V_out_avg, label='V_out')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('V_in and V_out vs Time')
plt.legend()

# Plot I_1 and I_Bat together
plt.subplot(2, 1, 2)
plt.plot(time_avg, I_1_avg, label='I_1')
plt.plot(time_avg, I_Bat_avg, label='I_Bat')
plt.xlabel('Time')
plt.ylabel('Current')
plt.title('I_1 and I_Bat vs Time')
plt.legend()

plt.tight_layout()
plt.savefig(f'{file_path[10:-4]}_Plot.png', dpi=1200)

