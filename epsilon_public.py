# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys 
import os
cwd = os.getcwd()
sys.path.append(os.path.abspath(cwd))
from myfunc_public import *
print (cwd)


# %%
# load and format data from source
cwd = os.getcwd()
data_path = cwd+'/Datasets/epsilon/epsilon_data.csv'
label_path = cwd+'/Datasets/epsilon/epsilon_labels.csv'
df = pd.read_csv(data_path, sep=',', header=None, chunksize=5500)
my_data = next(df)
print (my_data.shape)
df_labels = pd.read_csv(label_path, sep=',', header=None, chunksize=5500)
labels = next(df_labels)
y = labels.replace(-1, 0).squeeze()
print (y.value_counts())
percentage = y.value_counts()[1]/(y.value_counts()[1] + y.value_counts()[0]) *100
print ("percentage of majority: %f%%" % (percentage))


# %%
# RANDOMLY split the dataset
rs_val = 12 # random state value
X_train, X_test, y_train, y_test = train_test_split(my_data, y, test_size=0.2, random_state = rs_val)
# data manipulation
thin_dim = 1000
density = 1.0/thin_dim
transformer = random_projection.SparseRandomProjection(n_components=thin_dim, random_state=123)
my_data_thin = transformer.fit_transform(my_data)
print("my_data_thin" ,my_data_thin.shape)
X_train_thin, X_test_thin, y_train_thin, y_test_thin = train_test_split(my_data_thin, y, test_size = 0.2, random_state= rs_val)
# print result
print("my_data_thin" ,my_data_thin.shape)
print("my_data shape=", my_data.shape)
print("X_train shape=", X_train.shape)
print("X_test shape=", X_test.shape)
print("y_train shape=", y_train.shape)
print("y_test shape=", y_test.shape)
print("X_train_thin shape=", X_train_thin.shape)
print("X_test_thin shape=", X_test_thin.shape)
print("y_train_thin shape=", y_train_thin.shape)
print("y_test_thin shape=", y_test_thin.shape)
y_train.value_counts()


# %%
# MinMaxScaler scale the data to the [0,1] range
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_train)
X_train_scaled = min_max_scaler.transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)
X_train_scaled = pd.DataFrame.from_records(X_train_scaled)
X_test_scaled = pd.DataFrame.from_records(X_test_scaled)

min_max_scaler_thin = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_train_thin)
X_train_thin_scaled = min_max_scaler_thin.transform(X_train_thin)
X_test_thin_scaled = min_max_scaler_thin.transform(X_test_thin)
X_train_thin_scaled = pd.DataFrame.from_records(X_train_thin_scaled)
X_test_thin_scaled = pd.DataFrame.from_records(X_test_thin_scaled)


# %%
X_train_fixed = float2fix2float(X_train_scaled, 16, 12)
X_test_fixed = float2fix2float(X_test_scaled, 16, 12)
# they are still floating point but with reduce precison


# %%
# single model accuracy and training time measurement
_, fat_acc = evaluate_model_mlp(X_train_scaled, y_train, X_test_scaled, y_test, input_dim = 2000, verbose=0)



# %%
# Learning data generation with bagging
thin_dim = 1000
density = 'auto'
num_splits = 15
X_train_thin_sets, X_valid_thin_sets, X_test_thin_sets, y_train_thin_sets, y_valid_thin_sets = rp_data_gen_reverse(X_train_fixed, y_train, X_test_fixed, sample_portion=0.5, n_splits=num_splits, thin_dim=thin_dim, density=density)


# %%
# Individual model training and validation
scores, members = model_fitting(X_train_thin_sets, y_train_thin_sets, X_valid_thin_sets, y_valid_thin_sets, n_splits = num_splits,  verbose=0, classifier = 'mlp', input_dim= thin_dim, batch_size=100, num_epochs= 50)


# %%
ensemble_scoring_plot_mlp(members, X_test_thin_sets, y_test_thin, n_splits= num_splits)


# %%
marker = itertools.cycle(('x','^','+','*','s','1','P', 'X'))


# %%
# Dimension + sample size sweeping
thin_dim_min = 500
thin_dim_max = 1500
step = 200
portion_min = 0.2
portion_max = 0.8
portion_step = 0.2
density = 'auto'
ensemble_score_sets_sets, single_score_sets_sets = [], []
for portion_iter in np.arange(portion_min, portion_max, portion_step):
    ensemble_score_sets, single_score_sets = [], []
    ax = plt.subplot(111) # subplot position (1, 1, 1)
    for idx, dim_iter in enumerate(range(thin_dim_min, thin_dim_max, step)):
        X_train_thin_sets, X_valid_thin_sets, X_test_thin_sets, y_train_thin_sets, y_valid_thin_sets = rp_data_gen_reverse(X_train_fixed, y_train, X_test_fixed, sample_portion=portion_iter, n_splits=num_splits, thin_dim=dim_iter, density=density)
        # Individual model training and validation
        scores, members = model_fitting(X_train_thin_sets, y_train_thin_sets, X_valid_thin_sets, y_valid_thin_sets, n_splits = num_splits, verbose=False, classifier = 'mlp', input_dim=dim_iter, batch_size=100, num_epochs=50)
        # Ensemble model testing
        single_scores, ensemble_scores = ensemble_scoring_mlp(members, X_test_thin_sets, y_test_thin, n_splits=num_splits)
        ensemble_score_sets.append(ensemble_scores)
        single_score_sets.append(single_scores)
        x_axis = [i for i in range(1, num_splits+1)]
        # ax.plot(x_axis, single_score_sets[dim_iter], marker='o', linestyle='None')
        ax.plot(x_axis, ensemble_score_sets[idx], marker=next(marker), label='dim= {}'.format(dim_iter))
    ensemble_score_sets_sets.append(ensemble_score_sets)
    single_score_sets_sets.append(ensemble_score_sets)
    ax.plot([0, 15], [fat_acc, fat_acc], 'b:', label = 'single fat acc')
    # Shrink current axis's height by 10% on the bottom
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    title_str = "sample portion:" + str(round(portion_iter, 2))
    plt.title(title_str)
    plt.show()


# %%
# transform list to ndarray
result = np.asarray(ensemble_score_sets_sets)
# reduce to 2D array
result = result.reshape(20,15)
# re-format
df = df = pd.DataFrame(data=result,index=pd.MultiIndex.from_product([['portion={}'.format(i) for i in np.arange(portion_min,portion_max,portion_step)], ['dim{}'.format(i) for i in range(thin_dim_min,thin_dim_max, step)]]),
    columns=['en_acc {}'.format(i) for i in range(1, 16)]
)
cwd = os.getcwd()
file_name = cwd+'/results/epsilon_mlp_plotdataY.csv'
df.to_csv(file_name)
