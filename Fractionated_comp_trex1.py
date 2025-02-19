import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd



def extract_data(filename):
    with open(filename, 'r') as infile:
        # Skip the header lines
        infile.readline()
        infile.readline()

        # Initialize data structures
        data = []
        POI = []
        POI_nonnorm = []

        # Process each line in the file
        for line in infile:
            words = line.split()
            try:
                # Parse data from the line
                dose = float(words[0])
                poi_ref_norm = float(words[1]) if words[1] != 'nan' else np.nan
                poi_max_ref_norm = float(words[2]) if words[2] != 'nan' else np.nan

                # Append parsed values
                POI.append(poi_max_ref_norm)  # Only `POI Max Ref Norm` column
                POI_nonnorm.append(poi_ref_norm)
                data.append([dose, poi_ref_norm, poi_max_ref_norm])  # All columns

            except ValueError:
                print(f"Error converting line: {line.strip()}")

    # Return NumPy arrays for POI and full data
    return np.array(POI_nonnorm), np.array(data), np.array(POI),


def make_multidose_arrays2(array,data_0, data_2x4, data_2x8, data_3x8, idx=3):
    if idx == 3: #3 elements of each dose, except controll
        data_0.append(array[0:4])
        data_2x4.append(array[4:7])
        data_2x8.append(array[7:10])
        data_3x8.append(array[10:13])
    if idx == 2: #2 elements of each dose, except controll
        data_0.append(array[0:4])
        data_2x4.append(array[4:6])
        data_2x8.append(array[6:8])
        data_3x8.append(array[8:10])
    if idx == 1: #Strange experimental format, but manually works in this one case.
        data_0.append(array[0:3])
        data_2x4.append(array[3:5])
        data_2x8.append(array[5:7])
        data_3x8.append(array[7:10])

    # Flatten the appended lists to avoid ragged arrays
    data_0 = np.concatenate(data_0, axis=0) #if data_0 else np.array([])
    data_2x4 = np.concatenate(data_2x4, axis=0)# if data_2x4 else np.array([])
    data_2x8 = np.concatenate(data_2x8, axis=0)# if data_2x8 else np.array([])
    data_3x8 = np.concatenate(data_3x8, axis=0)# if data_3x8 else np.array([])

    return data_0, data_2x4, data_2x8, data_3x8

def make_multidose_arrays(array, data_0, data_2x4, data_2x8, data_3x8, idx=3):
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

    return data_0, data_2x4, data_2x8, data_3x8


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

def filter(data, type):
    data = np.array(data, dtype=float).flatten()  # Ensure 1D array
    if data.size == 0:
        return data  # Return empty array if input is empty
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
        threshold = 1.5
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
        print(data_0_M1F)
        print('--------- 2x4 Gy, MOC1--------------')
        print(data_2x4_M1F)
        print('--------- 2x8 Gy, MOC1--------------')
        print(data_2x8_M1F )
        print('--------- 3x8 Gy, MOC1--------------')
        print(data_3x8_M1F )
    if M >= 2:
        print('--------- 0 Gy, MOC2--------------')
        print(data_0_M2F)
        print('--------- 2x4 Gy, MOC2--------------')
        print(data_2x4_M2F)
        print('--------- 2x8 Gy, MOC2--------------')
        print(data_2x8_M2F)
        print('--------- 3x8 Gy, MOC2--------------')
        print(data_3x8_M1F)

def plot_seprate(data, type, idx = 3, norm_idx =2):
    dose = np.array([0, 4*2, 8*2, 8*3])
    S_data = data[:, norm_idx]

    if idx == 3:
        doses = np.array([0, 0, 0, 0, 4*2, 4*2, 4*2, 8*2, 8*2, 8*2, 8*3, 8*3, 8*3])
        mean = np.array([np.nanmean(S_data[0:4]), np.nanmean(S_data[4:7]), np.nanmean(S_data[7:10]), np.nanmean(S_data[10:13])])
    if idx == 2:
        doses = np.array([0, 0, 0, 0, 4*2, 4*2,  8*2, 8*2, 8*3, 8*3])
        mean = np.array([np.nanmean(S_data[0:4]), np.nanmean(S_data[4:6]), np.nanmean(S_data[6:8]), np.nanmean(S_data[8:10])])
    if idx == 1:
        doses = np.array([0, 0, 0, 4*2, 4*2,  8*2, 8*2, 8*3, 8*3,  8*3])
        mean = np.array([np.nanmean(S_data[0:3]), np.nanmean(S_data[3:5]), np.nanmean(S_data[5:7]), np.nanmean(S_data[7:10])])


    #mean = np.array([np.nanmean(S_data[0:4]), np.nanmean(S_data[4:7]), np.nanmean(S_data[7:10]), np.nanmean(S_data[10:13])])
    # Create scatter plot
    plt.scatter(doses, S_data, label = type)

    # Annotate each point with its index
    for i, (x, y) in enumerate(zip(doses, S_data)):
        plt.text(x, y, str(i), fontsize=8, ha='right', va='bottom')  # Position the label near the point

    # Add plot details
    ticks = ['0', '2 x 4Gy','2 x 8Gy', '3 x 8Gy' ]
    plt.xticks(dose, ticks)
    plt.title(f'{type} TREX1, scatter all non-nan data points. Control value normalization of data')
    plt.ylabel('TREX1 levels')
    plt.xlabel('Dose [Gy]')
    plt.plot(dose, mean, color="orange")
    plt.bar(dose, mean, color="orange", alpha=0.10)
    plt.grid(True)
    plt.legend()
    #plt.show()



def plot_mean_combined(doses, a0, a4, a8, a12, SEM, type, colour, norm_idx =1):
    mean = np.array([np.nanmean(a0), np.nanmean(a4),np.nanmean(a8), np.nanmean(a12)])
    plt.scatter(np.zeros(len(a0)), a0, label=f'0 Gy, Mean value: {mean[0]:.2f}, SEM: {SEM[0]:.2f} ')#, color=colour)
    plt.scatter(np.ones(len(a4)) * 4*2, a4, label=f'4 Gy, Mean value: {mean[1]:.2f}, SEM: {SEM[1]:.2f}')#, color=colour)
    plt.scatter(np.ones(len(a8)) * 8*2, a8, label=f'8 Gy, Mean value: {mean[2]:.2f}, SEM: {SEM[2]:.2f}')#, color=colour)
    plt.scatter(np.ones(len(a12)) * 8*3, a12, label=f'12 Gy, Mean value: {mean[3]:.2f}, SEM: {SEM[3]:.2f}')#, color=colour)
    #np.nanmean(a5)
    #title = f'{type[0]}: TREX1 levels of fractionated irridiation regime. Mean barplot, shaded SEM region with errorbars and scatter of all non nan data points. Controll value normalized and {type[1]} filtered.'
    title = f'{type[0]}: TREX1 levels of fractionated irridiation regime. Mean barplot, shaded SEM region with errorbars and scatter of all non nan data points. {type[1]} filtered. NOT dose controll value normalized'

    plt.title(title, loc='center', wrap=True)
    ticks = ['0', '2 x 4Gy','2 x 8Gy', '3 x 8Gy' ]
    plt.xticks(doses, ticks)
    plt.ylabel('TREX1 levels')
    plt.xlabel('Dose [Gy]')
    plt.errorbar(doses, mean, SEM, fmt='.', capsize=5, color='orange')
    plt.plot(doses, mean, color="orange")
    plt.fill_between(doses, mean-SEM, mean+SEM, alpha=0.3, color='yellow', label = 'SEM')
    plt.bar(doses, mean, color="orange", alpha=0.50)
    plt.margins(x=0, y=0)
    plt.grid(True)
    plt.legend()
    #plt.show()

#MOC1

filename = str('WESTERN22_gel2.txt')
data22F_1, data, a= extract_data(filename)
plot_seprate(data, 'MOC1, exp. 22F E3', 3)

filename = str('WESTERN23F_gel1.txt')
data23F_1, data, a = extract_data(filename)
plot_seprate(data, 'MOC1, exp. 23F E1', 3)

filename = str('WESTERN23F_gel3.txt')
data23F_3, data, a = extract_data(filename)
plot_seprate(data, 'MOC1, exp. 23F E2', 3)

filename = str('WESTERN24F_gel2.txt')
data24F_2, data, a = extract_data(filename)
plot_seprate(data, 'MOC1, exp. 24F E2', 2)

filename = str('WESTERN24F_gel3.txt')
data24F_3, data, a = extract_data(filename)
plot_seprate(data, 'MOC1, exp. 24F E1', 2)
plt.show()


array = np.array(data22F_1)  # Ensure `array` is a NumPy array
if len(array) < 13:
    raise ValueError("Array must have at least 13 elements for slicing.")

Fmet = 'IQR'# 'Z-score' or 'IQR'

data_0_M1 =[]; data_2x4_M1=[]; data_2x8_M1=[]; data_3x8_M1=[]
data_0_M1,data_2x4_M1, data_2x8_M1, data_3x8_M1 = make_multidose_arrays(data22F_1, data_0_M1, data_2x4_M1, data_2x8_M1, data_3x8_M1)
data_0_M1,data_2x4_M1, data_2x8_M1, data_3x8_M1 = make_multidose_arrays(data23F_1, data_0_M1, data_2x4_M1, data_2x8_M1, data_3x8_M1)
data_0_M1,data_2x4_M1, data_2x8_M1, data_3x8_M1 = make_multidose_arrays(data23F_3, data_0_M1, data_2x4_M1, data_2x8_M1, data_3x8_M1)
data_0_M1,data_2x4_M1, data_2x8_M1, data_3x8_M1 = make_multidose_arrays(data24F_2, data_0_M1, data_2x4_M1, data_2x8_M1, data_3x8_M1, idx = 2)
data_0_M1,data_2x4_M1, data_2x8_M1, data_3x8_M1 = make_multidose_arrays(data24F_3, data_0_M1, data_2x4_M1, data_2x8_M1, data_3x8_M1, idx = 2)


data_0_M1, data_2x4_M1, data_2x8_M1, data_3x8_M1 = np.array(data_0_M1), np.array(data_2x4_M1),  np.array(data_2x8_M1),  np.array(data_3x8_M1)
data_0_M1F, data_2x4_M1F, data_2x8_M1F, data_3x8_M1F = filter(data_0_M1, Fmet), filter(data_2x4_M1, Fmet), filter(data_2x8_M1, Fmet), filter(data_3x8_M1, Fmet)

#MOC2
filename = str('WESTERN23F_gel2.txt')
data23F_2,data, a = extract_data(filename)
#plot_seprate(data, 'MOC2, exp. 23F E1', 3)

filename = str('WESTERN23F_gel4.txt')
data23F_4,data, a= extract_data(filename)
#plot_seprate(data, 'MOC2, exp. 23F E2', 3)

filename = str('WESTERN24F_gel1.txt')
data24F_1, data, a = extract_data(filename)
#plot_seprate(data, 'MOC2, exp. 24F E3', 1)

filename = str('WESTERN24F_gel4.txt')
data24F_4, data, a = extract_data(filename)
#plot_seprate(data, 'MOC2, exp. 24F E1', 2)

#plt.show()

data_0_M2 =[]; data_2x4_M2=[]; data_2x8_M2=[]; data_3x8_M2=[]
data_0_M2,data_2x4_M2, data_2x8_M2, data_3x8_M2= make_multidose_arrays(data23F_2,data_0_M2, data_2x4_M2, data_2x8_M2, data_3x8_M2)
data_0_M2,data_2x4_M2, data_2x8_M2, data_3x8_M2= make_multidose_arrays(data23F_4,data_0_M2, data_2x4_M2, data_2x8_M2, data_3x8_M2)
data_0_M2,data_2x4_M2, data_2x8_M2, data_3x8_M2= make_multidose_arrays(data24F_1,data_0_M2, data_2x4_M2, data_2x8_M2, data_3x8_M2, idx = 1)
data_0_M2,data_2x4_M2, data_2x8_M2, data_3x8_M2= make_multidose_arrays(data24F_4,data_0_M2, data_2x4_M2, data_2x8_M2, data_3x8_M2, idx = 2)

data_0_M2,data_2x4_M2, data_2x8_M2, data_3x8_M2 = np.array(data_0_M2), np.array(data_2x4_M2),  np.array(data_2x8_M2),  np.array(data_3x8_M2)
data_0_M2F, data_2x4_M2F, data_2x8_M2F, data_3x8_M2F = filter(data_0_M2, Fmet), filter(data_2x4_M2, Fmet), filter(data_2x8_M2, Fmet), filter(data_3x8_M2, Fmet)

#Create plots.
doses_multi = np.array([0, 4*2, 8*2, 8*3])
SEM_M1 = SEM(data_0_M1F, data_2x4_M1F, data_2x8_M1F,data_3x8_M1F)
plot_mean_combined(doses_multi, data_0_M1F, data_2x4_M1F, data_2x8_M1F,data_3x8_M1F, SEM_M1, ['MOC1', Fmet] , 'blue')
plt.show()
print_values(1)

# Perform ANOVA
f_stat, p_value = perform_anova( data_0_M1F, data_2x4_M1F, data_2x8_M1F,data_3x8_M1F)
print(f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}")

# Perform Tukey's HSD post-hoc analysis
tukey_results = perform_posthoc_tukey(doses_multi,data_0_M1F, data_2x4_M1F, data_2x8_M1F,data_3x8_M1F)
print(tukey_results)

SEM_M2 = SEM(data_0_M2F, data_2x4_M2F, data_2x8_M2F,data_3x8_M2F)
plot_mean_combined(doses_multi, data_0_M2F, data_2x4_M2F, data_2x8_M2F,data_3x8_M2F, SEM_M2, ['MOC2', Fmet], 'red')
plt.show()
print_values(2)

# Perform ANOVA
f_stat, p_value = perform_anova( data_0_M2F, data_2x4_M2F, data_2x8_M2F,data_3x8_M2F,)
print(f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}")

# Perform Tukey's HSD post-hoc analysis
tukey_results = perform_posthoc_tukey(doses_multi, data_0_M2F, data_2x4_M2F, data_2x8_M2F,data_3x8_M2F,)
print(tukey_results)
