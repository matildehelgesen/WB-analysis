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

def make_dose_arrays(idx1, idx2, type):
    a1 = data17_1[idx1,type]; a2 = data17_1[idx2,type]
    a3 = data17_2[idx1,type]; a4 = data17_2[idx2,type]
    a5 = data16_1[idx1,type]; a6 = data16_1[idx2,type]
    a7 = data16_2[idx1,type]; a8 = data16_2[idx2,type]
    a9 = data14_1[idx1,type]; a10 = data14_1[idx2,type]
    data = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10])
    return data

def SEM1(array, n=10, N=4):#n = nr of experiments accounted for.
    factor = 1/(n*(n-1))
    sum = np.zeros(N)
    for j in range(N):
        mean = np.nanmean(array[j, :])
        for i in range(n):
            if not np.isnan(array[j, i]):  # Ignore nan values
                sum += (array[j, i] - mean) ** 2
    SEM = np.sqrt(factor*sum)
    return SEM

def SEM(*arrays):#n = nr of experiments accounted for.
    sem_values = []
    for array in arrays:
        n = len(array)
        if n > 1:
            sem = np.nanstd(array) / np.sqrt(n)  # Standard error of the mean
        else:
            sem = np.nan  # If not enough data points, return NaN
        sem_values.append(sem)

    return np.array(sem_values)

def plot_dataMax(data, title, legend1):
    #plt.plot( data[0],data[2], m',label = 'Time response II')
    plt.scatter(data[:,0],data[:,2], label = legend1)
    plt.title(title)
    plt.ylabel('HMGB1 levels')
    plt.xlabel('dose[Gy]')
    plt.bar(data[:,0],data[:,2], color="orange", alpha = 0.25)
    plt.grid(True)
    plt.legend()

def plot_dataref(data, title, legend1):
    plt.scatter(data[:,0],data[:,1],label = legend1)
    plt.title(title)
    plt.ylabel('HMGB1 levels')
    plt.xlabel('dose[Gy]')
    plt.bar(data[:,0],data[:,1], color="orange", alpha = 0.25)
    plt.grid()
    plt.legend()
    plt.show()

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





def plot_mean(doses, a0, a0_, a4, a4_, a8, a8_,  a12, a12_, SEM, title, p_values):
    mean = np.array([np.nanmean(a0), np.nanmean(a4),np.nanmean(a8),np.nanmean(a12)])
    plt.scatter(np.zeros(len(a0_)) ,a0_, label = f'0Gy, Mean value: {mean[0]:.2f}, SEM: {SEM[0]:.2f}')
    plt.scatter(np.ones(len(a4_))*4 ,a4_, label = f'4Gy, Mean value:{mean[1]:.2f}, SEM:  {SEM[1]:.2f}')
    plt.scatter(np.ones(len(a8_))*8 ,a8_, label = f'8Gy, Mean value: {mean[2]:.2f}, SEM: {SEM[2]:.2f}')
    plt.scatter(np.ones(len(a12_))*12,a12_, label = f'12Gy, Mean value: {mean[3]:.2f}, SEM: {SEM[3]:.2f}')


    plt.title(title, loc='center', wrap=True)
    plt.ylabel('HMGB1 levels')
    plt.xlabel('Dose[Gy]')
    plt.errorbar(doses, mean, SEM, fmt='.', capsize=5)
    plt.plot(doses, mean, color="orange", )
    plt.bar(doses, mean, color="orange", alpha = 0.5)
    plt.fill_between(doses, mean-SEM, mean+SEM, alpha=0.3, color='yellow', label = 'SEM')
    # Annotate plot with p-values
   # Annotate plot with p-values
    for (dose1, dose2), p_value in p_values.items():
        if p_value < 0.05:  # Only annotate significant results
            x_pos = (dose1 + dose2) / 2
            y_pos = max(mean[doses == dose1], mean[doses == dose2]) + 0.5  # Adjust y-position
            plt.text(x_pos, y_pos, f'p={p_value:.4f}', ha='center', fontsize=9)


    plt.grid(True)
    plt.legend()
    plt.show()



filename = str('WESTERN17_gel11.txt')
data17_1 = extract_data(filename)
#plot_dataMax(data17_2, filename,'W17 g2')

filename = str('WESTERN17_gel21.txt')
data17_2 = extract_data(filename)
#plot_dataMax(data17_2, filename,'W17 g2')

filename = str('WESTERN16_gel11.txt')
data16_1 = extract_data(filename)
#plot_dataMax(data16_1, filename,'W16 g1')

filename = str('WESTERN16_gel21.txt')
data16_2 = extract_data(filename)
#plot_dataMax(data16_2, filename, 'W16 g2')

filename = str('WESTERN14_gel1.txt')
data14_1 = extract_data(filename)
#plot_dataMax(data14_1, filename, 'W14 g1')
#plt.show()

""" refrence normalized refrence signal normalization of data"""
doses = np.array([0,4,8,12])
norm_idx= 2 # control= 2 , ref = 1
Fmet = 'IQR'# 'Z-score' or 'IQR'

data_0gy = make_dose_arrays(0, 1, norm_idx)
data_0gy_f = filter(data_0gy, Fmet)
data_4gy = make_dose_arrays(2, 3, norm_idx)
data_4gy_f = filter(data_4gy, Fmet)
data_8gy = make_dose_arrays(4, 5, norm_idx)
data_8gy_f = filter(data_8gy, Fmet)
data_12gy = make_dose_arrays(6, 7, norm_idx)
data_12gy_f = filter(data_12gy, Fmet)

print('------- 0 -------')
print(data_0gy)
print(data_0gy_f)
print('------- 4 -------')
print(data_4gy)
print(data_4gy_f)
print('------- 8 -------')
print(data_8gy)
print(data_8gy_f)
print('------- 12 -------')
print(data_12gy)
print(data_12gy_f)


# Perform t-tests
p_values = perform_t_tests(doses, data_0gy_f, data_4gy_f, data_8gy_f, data_12gy_f)


#all_data = np.array(data_0gy_f, data_4gy_f ,data_8gy_f, data_12gy_f)
Title = f'MOC 1: HMGB1 levels. Mean barplot, shaded SEM region with errorbars and scatter of all non nan data points. Controll dose normalized and {Fmet} filtered.'
SEMr = SEM(data_0gy_f, data_4gy_f ,data_8gy_f, data_12gy_f)
plot_mean(doses, data_0gy_f,data_0gy, data_4gy_f,data_4gy, data_8gy_f, data_8gy, data_12gy_f,data_12gy, SEMr, Title, p_values)
plt.show()

#non filtered_data
Title = f'MOC 1: HMGB1 levels. Mean and SEM barplot and scatter of all non nan data points. Controll dose normalized and non filtered data'
SEMr = SEM(data_0gy, data_4gy ,data_8gy, data_12gy)
plot_mean(doses, data_0gy_f,data_0gy, data_4gy_f,data_4gy, data_8gy_f, data_8gy, data_12gy_f,data_12gy, SEMr, Title, p_values)
plt.show()


# Define doses and filter data
doses = np.array([0, 4, 8, 12])
Fmet = 'IQR'  # Filtering method

data_0gy = make_dose_arrays(0, 1, 2)
data_0gy_f = filter(data_0gy, Fmet)
data_4gy = make_dose_arrays(2, 3, 2)
data_4gy_f = filter(data_4gy, Fmet)
data_8gy = make_dose_arrays(4, 5, 2)
data_8gy_f = filter(data_8gy, Fmet)
data_12gy = make_dose_arrays(6, 7, 2)
data_12gy_f = filter(data_12gy, Fmet)

# Calculate SEM
SEMr = SEM(data_0gy_f, data_4gy_f, data_8gy_f, data_12gy_f)

# Perform ANOVA
# Perform ANOVA
f_stat, p_value = perform_anova(data_0gy_f, data_4gy_f, data_8gy_f, data_12gy_f)
print(f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}")

# Perform Tukey's HSD post-hoc analysis
tukey_results = perform_posthoc_tukey(doses, data_0gy_f, data_4gy_f, data_8gy_f, data_12gy_f)
print(tukey_results)
