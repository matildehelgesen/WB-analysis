import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd



def extract_data(filename):
    infile = open(filename, 'r')
    infile.readline()
    infile.readline()

    no_lines = len(open(filename).readlines())
    data = []     # Read each line, split the words, and convert to floats
    for line in infile:
        words = line.split()
        try:
            # Attempt to convert values; use np.nan for invalid floats
            dose = float(words[0])
            poi_ref_norm = float(words[1]) if words[1] != 'nan' else np.nan
            poi_max_ref_norm = float(words[2]) if words[2] != 'nan' else np.nan
            data.append([dose, poi_ref_norm, poi_max_ref_norm])
        except ValueError:
            # Handle the case where conversion fails
            print(f"Error converting line: {line.strip()}")

    return np.array(data)

def make_multidose_arrays(array, data_0, data_4, data_8, data_12, idx=3):
    if idx == 3:  # 3 elements of each dose, except control
        data_0 += list(array[0:4])
        data_2x4 += list(array[4:7])
        data_2x8 += list(array[7:10])
        data_3x8 += list(array[10:13])
    elif idx == 2:  # 2 elements of each dose, except control
        data_0 += list(array[0:4])
        data_2x4 += list(array[4:6])
        data_2x8 += list(array[6:8])
        data_3x8 += list(array[8:10])
    elif idx == 1:  # Custom experimental format
        data_0 += list(array[0:3])
        data_2x4 += list(array[3:5])
        data_2x8 += list(array[5:7])
        data_3x8 += list(array[7:10])

    return data_0, data_4, data_8, data_12


def SEM(*arrays):
    sem_values = []
    for array in arrays:
        n = len(array)
        if n > 1:
            sem = np.nanstd(array) / np.sqrt(n)  # Standard error of the mean
        else:
            sem = np.nan  # If not enough data points, return NaN
        sem_values.append(sem)

    return np.array(sem_values)

def filter(data, type):
    if type == 'IQR':
        # Calculate Q1, Q3, and IQR
        std_dev = np.nanstd(data)
        Q1 = np.nanpercentile(data, 25)
        Q3 = np.nanpercentile(data, 75)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    elif type == 'Z-score':
        mean = np.nanmean(data)
        std_dev = np.nanstd(data)
        z_scores = (data - mean) / std_dev

        # Define a threshold for outliers, e.g., Z > 3
        threshold = 3
        filtered_data = data[np.abs(z_scores) < threshold]

    return filtered_data


def perform_t_tests(doses, *groups):
    """
    Perform pairwise t-tests between dose groups.

    Parameters:
    - doses: List of dose levels.
    - groups: Data arrays for each dose group.

    Returns:
    - p_values: A dictionary of p-values with keys as dose pairs.
    """
    p_values = {}
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if j > i:  # Avoid duplicate comparisons
                t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
                p_values[(doses[i], doses[j])] = p_value
    return p_values

def perform_anova(*groups):
        """
        Perform one-way ANOVA across multiple dose groups.

        Parameters:
        - groups: Arrays containing data for each dose group.

        Returns:
        - f_stat: F-statistic of the ANOVA.
        - p_value: p-value of the ANOVA.
        """
        f_stat, p_value = f_oneway(*groups)
        return f_stat, p_value


def perform_posthoc_tukey(doses, *groups):
    """
    Perform Tukey's HSD post-hoc test for multiple comparisons.

    Parameters:
    - doses: List of dose levels.
    - groups: Data arrays for each dose group.

    Returns:
    - tukey_results: Tukey HSD results object.
    """
    # Flatten data and group labels for Tukey's test
    flattened_data = []
    group_labels = []
    for i, group in enumerate(groups):
        flattened_data.extend(group)
        group_labels.extend([doses[i]] * len(group))

    # Perform Tukey's HSD test
    tukey_results = pairwise_tukeyhsd(endog=flattened_data, groups=group_labels, alpha=0.05)
    return tukey_results



def print_values(M):
    if M <= 1:
        print('--------- 0 Gy, MOC1--------------')
        print(data_0gy_M1 )
        print('--------- 4 Gy, MOC1--------------')
        print(data_4gy_M1 )
        print('--------- 8 Gy, MOC1--------------')
        print(data_8gy_M1 )
        print('--------- 12 Gy, MOC1--------------')
        print(data_12gy_M1 )
    if M >= 2:
        print('--------- 0 Gy, MOC2--------------')
        print(data_0gy_M2)
        print('--------- 4 Gy, MOC2--------------')
        print(data_4gy_M2 )
        print('--------- 8 Gy, MOC2--------------')
        print(data_8gy_M2 )
        print('--------- 12 Gy, MOC2--------------')
        print(data_12gy_M2 )

def plot_seprate(data, type, idx, norm_idx=2):
    doses = data[:,0]
    dose = np.array([0,4,8,12])
    S_data = data[:,2]
    if idx == 3:#to be able to handle experiments with 2 and 3 samples pper dose
        mean = np.array([np.nanmean(S_data[0:3]), np.nanmean(S_data[3:6]) , np.nanmean(S_data[6:9]), np.nanmean(S_data[9:12])])
    if idx == 2:#to be able to handle experiments with 2 and 3 samples pper dose
        mean = np.array([np.nanmean(S_data[0:2]), np.nanmean(S_data[2:4]) , np.nanmean(S_data[4:6]), np.nanmean(S_data[6:8])])

    plt.scatter(doses,S_data, label = type)
    if norm_idx == 2:
            plt.title(f'{type} TREX1, scatter all non nan data points. controll value normalization of data')
    plt.ylabel('TREX1 levels')
    plt.xlabel('Dose [Gy]')
    #plt.plot(dose, mean,label = type, color="orange")
    #plt.bar(dose, mean, color="orange", alpha=0.50)
    plt.grid(True)
    #plt.legend()
    #plt.show()



def plot_mean_combined(doses, a0, a4, a8, a12, SEM, norm_idx, type, colour):
    mean = np.array([np.nanmean(a0), np.nanmean(a4),np.nanmean(a8),np.nanmean(a12)])
    plt.scatter(np.zeros(len(a0)) ,a0, label = f'0Gy , Mean value: {mean[0]:.5f}, SEM: {SEM[0]:.2f}')
    plt.scatter(np.ones(len(a4))*4 ,a4, label = f'4Gy , Mean value:{mean[1]:.5f}, SEM:  {SEM[1]:.2f}')
    plt.scatter(np.ones(len(a8))*8 ,a8, label = f'8Gy, Mean value: {mean[2]:.5f}, SEM: {SEM[2]:.2f}')
    plt.scatter(np.ones(len(a12))*12,a12, label = f'12Gy, Mean value: {mean[3]:.5f}, SEM: {SEM[3]:.2f}')

    if norm_idx ==1:
        title = (f'{type} TREX1,  Mean and SEM, scatter all non nan data points. refrence signal normalization of data')
    if norm_idx == 2:
        title = (f'{type[0]}: TREX1 levels. Mean barplot, shaded SEM region with errorbars and scatter of all non nan data points. Controll dose normalized and {type[1]} filtered.')
    plt.title(title, loc='center', wrap=True)
    plt.ylabel('TREX1 levels')
    plt.xlabel('Dose [Gy]')
    plt.errorbar(doses, mean, SEM, fmt='.', capsize=5, color='orange')
    plt.plot(doses, mean, color="orange")
    plt.fill_between(doses, mean-SEM, mean+SEM, alpha=0.3, color='yellow', label = 'SEM', linestyle='dashdot')
    plt.bar(doses, mean, color="orange", alpha=0.50)
    plt.grid(True)
    plt.legend()
    #plt.show()


filename = str('WESTERNp1_gel1.txt')
datap1_M1 = extract_data(filename)
filename = str('WESTERNp1_gel2.txt')
datap1_M2 = extract_data(filename)




"""
plot_seprate(data19_1_M1, 'MOC1, exp. 19', 2)
plot_seprate(data20_1_M1, 'MOC1, exp. 20', 2)
plot_seprate(data21_2_M1, 'MOC1, exp. 21', 3)
plt.legend()
plt.show()

plot_seprate(data19_2_M2, 'MOC2, exp. 19', 2)
plot_seprate(data20_2_M2, 'MOC2, exp. 20', 2)
plot_seprate(data21_1_M2, 'MOC2, exp. 21', 3)
plt.legend()
plt.show()
"""


norm_idx= 1 # control= 2 , ref = 1


""" FILTERED USING IQRfilter"""
Fmet = 'IQR'# 'Z-score' or 'IQR'
doses_multi = np.array([0, 4, 8, 12])
""" MOC1 norm_idx normalized refrence signal normalization of data"""
data_0gy_M1_ = make_multidose_arrays_1(0, 0, norm_idx)
data_0gy_M1= filter(data_0gy_M1_, Fmet)
data_4gy_M1_ = make_multidose_arrays_1(2, 3, norm_idx)
data_4gy_M1 =filter(data_4gy_M1_, Fmet)
data_8gy_M1_ = make_multidose_arrays_1(4, 6, norm_idx)
data_8gy_M1 =filter(data_8gy_M1_, Fmet)
data_12gy_M1_ = make_multidose_arrays_1(6, 9, norm_idx)
data_12gy_M1 = filter(data_12gy_M1_, Fmet)
all_data_M1 = np.concatenate([data_0gy_M1, data_4gy_M1, data_8gy_M1, data_12gy_M1])
print_values(0)

SEM_M1 = SEM(data_0gy_M1, data_4gy_M1, data_8gy_M1, data_12gy_M1)
plot_mean_combined(doses_multi, data_0gy_M1, data_4gy_M1, data_8gy_M1, data_12gy_M1, SEM_M1, norm_idx, ['MOC1', Fmet], 'blue')
plt.show()

# Perform ANOVA
f_stat, p_value = perform_anova(data_0gy_M1, data_4gy_M1, data_8gy_M1, data_12gy_M1)
print(f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}")

# Perform Tukey's HSD post-hoc analysis
tukey_results = perform_posthoc_tukey(doses_multi, data_0gy_M1, data_4gy_M1, data_8gy_M1, data_12gy_M1)
print(tukey_results)



""" MOC2 norm_idx normalized refrence signal normalization of data"""
data_0gy_M2_ = make_multidose_arrays_2(0, 0, norm_idx)
data_0gy_M2= filter(data_0gy_M2_, Fmet)
data_4gy_M2_ = make_multidose_arrays_2(2, 3, norm_idx)
data_4gy_M2 =filter(data_4gy_M2_, Fmet)
data_8gy_M2_ = make_multidose_arrays_2(4, 6, norm_idx)
data_8gy_M2 =filter(data_8gy_M2_, Fmet)
data_12gy_M2_ = make_multidose_arrays_2(6, 9, norm_idx)
data_12gy_M2 = filter(data_12gy_M2_, Fmet)
all_data_M2 = np.concatenate([data_0gy_M2, data_4gy_M2, data_8gy_M2, data_12gy_M2])

SEM_M2= SEM(data_0gy_M2, data_4gy_M2, data_8gy_M2, data_12gy_M2)
plot_mean_combined(doses_multi, data_0gy_M2, data_4gy_M2, data_8gy_M2, data_12gy_M2, SEM_M2, norm_idx, ['MOC2', Fmet], 'red')
print_values(2)
plt.show()

# Perform ANOVA
f_stat, p_value = perform_anova(data_0gy_M2, data_4gy_M2, data_8gy_M2, data_12gy_M2)
print(f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}")

# Perform Tukey's HSD post-hoc analysis
tukey_results = perform_posthoc_tukey(doses_multi, data_0gy_M2, data_4gy_M2, data_8gy_M2, data_12gy_M2)
print(tukey_results)
