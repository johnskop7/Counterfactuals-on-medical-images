#piece.py
import numpy as np
import scipy.stats as stats
import torch
import pickle
from tqdm import tqdm
import pandas as pd



class HurdleModel:
    """
    A statistical hurdle model to find the probability of extracted features at layer X in a CNN for the counterfactual class c'
    """

    def __init__(self, data, value, p_value):
        if np.sum(data) < 0:
            print("*** ERROR: Must be Positive Data ***")

        self.data = np.array(data, dtype=float)
        if len(self.data) == 0 or np.isnan(self.data).all():
            raise ValueError("Data is empty or all values are NaN.")

        if np.sum(self.data) < 0:
            raise ValueError("*** ERROR: Must be Positive Data ***")

        self.value = float(value)
        self.filtered_data = self.data[self.data != 0]
        self.rv, fixed_location = self.__get_dist_type()

        # Try all six PDF options
        if fixed_location:
            self.params = self.rv.fit(self.filtered_data, floc=0)
        else:
            self.params = self.rv.fit(self.filtered_data)

        self.fixed_location = fixed_location
        self.p_value = p_value
        self.bern_param = len(self.filtered_data) / len(self.data)  # probability of "success" in Bernoulli trial

    def __get_dist_type(self):
        p_values = {'norm none': None, 'gamma none': None, 'expon none': None,
                    'norm floc': None, 'gamma floc': None, 'expon floc': None}

        for test in ['norm', 'gamma', 'expon']:
            for location in ['none', 'floc']:
                if test == 'norm':
                    rv = stats.norm
                elif test == 'gamma':
                    rv = stats.gamma
                elif test == 'expon':
                    rv = stats.expon

                if location == 'none':
                    params = rv.fit(self.filtered_data)
                elif location == 'floc':
                    params = rv.fit(self.filtered_data, floc=0)

                p_values[test + " " + location] = (stats.kstest(self.filtered_data, test, args=params)[1])

        max_key = max(p_values, key=lambda k: p_values[k])
        dist_type, location = max_key.split(" ")

        if dist_type == 'norm':
            if location == 'none':
                return stats.norm, False
            elif location == 'floc':
                return stats.norm, True
        if dist_type == 'gamma':
            if location == 'none':
                return stats.gamma, False
            elif location == 'floc':
                return stats.gamma, True
        if dist_type == 'expon':
            if location == 'none':
                return stats.expon, False
            elif location == 'floc':
                return stats.expon, True

    def __pdf(self, x):
        return self.rv.pdf(x, *self.params) * self.bern_param

    def __cdf(self, x):
        return self.rv.cdf(x, *self.params) * self.bern_param

    def get_cdf(self, x):
        return self.rv.cdf(x, *self.params) * self.bern_param

    def __ppf_upper_sig_value(self):
        return self.rv.ppf(0.999, *self.params) * self.bern_param

    def __ppf_lower_sig_value(self):
        return self.rv.ppf(0.001, *self.params) * self.bern_param

    def get_expected_value(self):
        return self.rv.mean(*self.params)

    def get_prob_of_value(self):
        if self.value == 0:
            return 1 - self.bern_param
        else:
            lower = self.__cdf(self.value)
            upper = self.bern_param - self.__cdf(self.value)
            return min(lower, upper)

    def bern_fail_sig(self):
        if (1 - self.bern_param) < self.p_value and self.value == 0:
            return True
        return False

    def bern_success_sig(self):
        if self.bern_param < (self.p_value * 2) and self.value > 0:
            return True
        return False

    def high_cont_sig(self):
        if (1 - self.bern_param) + self.__cdf(self.value) > (1 - self.p_value):
            return True
        return False

    def low_cont_sig(self):
        if self.__cdf(self.value) < self.p_value and self.value != 0:
            return True
        return False

    def test_fit(self):
        return stats.kstest(self.filtered_data, self.rv.name, args=self.params)[1]
    

def get_data_for_feature(dist_data, target_class, feature_map_num):
    data = np.array(dist_data[target_class][:])
    data = data.T[feature_map_num].T.reshape(data.shape[0], 1)
    return data

def get_distribution_name(dist):
    if dist.fixed_location == True:
        return dist.rv.name + " With Fixed 0 Location"
    else:
        return dist.rv.name

def acquire_feature_probabilities(target_class, cnn, original_query_img=None, alpha=0.05 , class_activations_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn.to(device)

    if original_query_img is not None:
        original_query_img = original_query_img.to(device)

    query_features = cnn(original_query_img)[1][0].detach().to(device)
    #digit_weights = cnn.classifier[2].weight
    digit_weights = cnn.model.fc.weight.detach().to(device)  # Get the weights of the fully connected layer
    #digit_weights = cnn.resnet101.fc.weight.data.detach().to(device)
    #digit_weights = cnn.alexnet.classifier[6].weight.data.detach().to(device) #this is for alexnet
    digit_weights = digit_weights[target_class]




    with open(class_activations_path, 'rb') as handle:
        dist_data = pickle.load(handle)

    fail_results = list()
    succeed_results = list()
    high_results = list()
    low_results = list()
    expected_values = list()
    probability = list()
    p_values = list()
    distribution_type = list()

    for i in tqdm(range(len(query_features)), desc="Processing Features"):
        data = get_data_for_feature(dist_data, target_class, feature_map_num=i)
        # Troubleshooting: Check the content of data
        try:
            data = np.array(data, dtype=float).reshape(-1, 1)
            feature_value = float(query_features[i])

            dist_examine = HurdleModel(data, value=feature_value, p_value=alpha)
            fail_results.append(dist_examine.bern_fail_sig())
            succeed_results.append(dist_examine.bern_success_sig())
            high_results.append(dist_examine.high_cont_sig())
            low_results.append(dist_examine.low_cont_sig())
            expected_values.append(dist_examine.get_expected_value())
            probability.append(dist_examine.get_prob_of_value())
            p_values.append(dist_examine.test_fit())
            distribution_type.append(get_distribution_name(dist_examine))
        except ValueError as e:
            print(f"Error processing feature {i}: {e}")
            fail_results.append(None)
            succeed_results.append(None)
            high_results.append(None)
            low_results.append(None)
            expected_values.append(None)
            probability.append(None)
            p_values.append(None)
            distribution_type.append(None)

    df = pd.DataFrame()
    df['Feature Map'] = list(range(len(query_features)))
    df['Contribution'] = query_features.detach().cpu().numpy() * digit_weights.detach().cpu().numpy()
    df['Bern Fail'] = fail_results
    df['Bern Success'] = succeed_results
    df['Cont High'] = high_results
    df['Cont Low'] = low_results
    df['Expected Value'] = expected_values
    df['Probability of Event'] = probability
    df['Distribution p-value KsTest'] = p_values
    df['Dist Type'] = distribution_type

    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    return df

def get_distribution_name(hurdle_model):
    return hurdle_model.rv.name



def filter_df_of_exceptional_noise(df, target_class, cnn, alpha=0.05):
    """
    Take the DataFrame, and remove rows which are exceptional features in c' (counterfactual class) but not candidate for change.
    Return: dataframe with only relevant features for PIECE algorithm

    Alpha is the probability threshold for what is "exceptional" or "weird" in the image.
    """
    df = df[df['Probability of Event'] < alpha].copy()  # Use .copy() to avoid the warning
    df['flag'] = np.zeros(df.shape[0])
    #digit_weights = cnn.classifier[2].weight #this is for convnext
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit_weights = cnn.model.fc.weight.detach().to(device)  # this is for resnet50
    #digit_weights = cnn.alexnet.classifier[6].weight.data.detach().to(device) #this is for alexnet
    digit_weights = digit_weights[target_class]


    for idx, row in df.iterrows():
        feature_idx = int(row['Feature Map'])
        cont = row['Contribution']
        cont_high = row['Cont High']
        cont_low = row['Cont Low']
        bern_fail = row['Bern Fail']
        expected_value = row['Expected Value']

        if bern_fail:  # if it's unusual to not activate, but it's negative
            if digit_weights[feature_idx] < 0:
                df.loc[idx, 'flag'] = 1
        if cont_high:  # if it's high, but positive
            if digit_weights[feature_idx] > 0:
                df.loc[idx, 'flag'] = 1
        if cont_low:  # if it's low, but negative
            if digit_weights[feature_idx] < 0:
                df.loc[idx, 'flag'] = 1

    df = df[df['flag'] == 0]
    df.drop('flag', axis=1, inplace=True)

    return df



def modifying_exceptional_features(df, target_class, query_activations):
    """
    Change all exceptional features to the expected value for each PDF
    return: tensor with all exceptional features turned into "expected" feature values for c'
    """

    ideal_xp = query_activations.clone().detach()
    for idx, row in df.sort_values('Probability of Event', ascending=True).iterrows():
        feature_idx = int(row['Feature Map'])
        if feature_idx >= ideal_xp.size(0):
            print(f"Skipping index {feature_idx} as it's out of bounds for the tensor size {ideal_xp.size(0)}")
            continue
        expected_value = row['Expected Value']
        ideal_xp[feature_idx] = expected_value

    return ideal_xp
    
