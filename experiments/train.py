import sys
import time
import numpy as np
import pandas as pd
sys.path.append('./')
from E2USD.e2usd import E2USD
from E2USD.adapers import *
from E2USD.utils import *
from E2USD.clustering import *
from E2USD.params import *
import pca
import matplotlib.pyplot as plt
import traceback
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TSpy.view import plot_mts, visualize_trace




# Define global paths
script_path = os.path.dirname(__file__)
data_path = os.path.join("C:\\Ipdu_stastics\\X2E_npy\\X2E_297_01096_20240318_130813_20240326_171322_no_someip\\")
output_path = os.path.join("C:\\Ipdu_stastics\\X2E_npy\\DP_HSMM")


def create_path(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def exp_on_trace(win_size, step):
    data_path = "C:\\Ipdu_stastics\\X2E_npy\\data.npy"
    data = np.load(data_path)
    time_path_data = "C:\\Ipdu_stastics\\X2E_npy\\pdu_time.npy"
    datatime = np.load(time_path_data)
    data_pca = pca.pca_process(data, 86)

    new_data_list = []
    new_data_list2 = []
    m = len(data_pca[0])
    n = len(data[0])
    for i in range(len(datatime) - 1):
        new_data_list.append(data_pca[i])
        new_data_list2.append(data[i])
        if datatime[i+1] - datatime[i] > 120:
            zero_block = np.zeros((300, m))
            new_data_list.append(zero_block)
            zero_block = np.zeros((300, n))
            new_data_list2.append(zero_block)


    new_data_list.append(data_pca[-1])

    data1 = np.vstack(new_data_list)

    new_data_list2.append(data[-1])

    data3 = np.vstack(new_data_list2)

    data2 = StandardScaler().fit_transform(data1)
    # overall_max_value = np.max(data )
    # data1  = data /overall_max_value

    # try:
    #     t2s.fit(data2, win_size, step)
    # except Exception as e:
    #     print("An exception occurred: ", str(e))
    #     traceback.print_exc()

   

    params['in_channels'] = 86
    params['out_channels'] = 30
    params['compared_length'] = win_size
    params['kernel_size'] = 3


    # dataset_path = data_path
    # fname = "pdu_data_segment_0.npy"
    # dataset_path = dataset_path + fname
    # X = np.load(dataset_path, allow_pickle=True)
    # data = np.vstack(X)
    # df = pd.DataFrame(data, columns=['Timestamp', 'Signal', 'Quantity1', 'Quantity2', 'Quantity3'])
    # df['Signal_Quantity1'] = df['Signal'] + '_' + df['Quantity1']
    # df_pivot = df.pivot_table(index='Timestamp', columns='Signal_Quantity1', values=['Quantity2'], aggfunc='first')
    # df_pivot = df_pivot.fillna(0)
    # result_array = df_pivot.to_numpy().astype(int)
    # data_pca = pca.pca_process(result_array, 18)
    t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data2, win_size, step)
    prediction = t2s.state_seq
    prediction = np.array(prediction, dtype=int)
    # np.save(os.path.join(out_path, fname), prediction)
    np.save("prediction.npy", prediction)
    plt.style.use('classic')
    visualize_trace(data3, prediction, min_zero_length=10)
    plot_mts(data3, prediction)
    plt.savefig('2noso.png')



# def exp_on_UCR_SEG(win_size, step, verbose=False):
#     """Experiment on UCR_SEG dataset."""
#     score_list = []
#     out_path = os.path.join(output_path, 'UCR-SEG')
#     create_path(out_path)
#     params['in_channels'] = 1
#     params['out_channels'] = 4
#     params['compared_length'] = win_size
#     params['kernel_size'] = 3


#     dataset_path = os.path.join(data_path, 'UCRSEG/')
#     for fname in os.listdir(dataset_path):
#         info_list = fname[:-4].split('_')
#         seg_info = {}
#         i = 0
#         for seg in info_list[2:]:
#             seg_info[int(seg)] = i
#             i += 1
#         seg_info[len_of_file(dataset_path + fname)] = i
#         df = pd.read_csv(dataset_path + fname)
#         data = df.to_numpy()
#         data = normalize(data)
#         t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data, win_size, step)
#         groundtruth = seg_to_label(seg_info)[:-1]
#         prediction = t2s.state_seq
#         ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
#         prediction = np.array(prediction, dtype=int)
#         result = np.vstack([groundtruth, prediction])
#         np.save(os.path.join(out_path, fname[:-4]), result)
#         score_list.append(np.array([ari, anmi, nmi]))
#         if verbose:
#             print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' % (fname, ari, anmi, nmi))
#     score_list = np.vstack(score_list)
#     print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
#                                                        , np.mean(score_list[:, 1])
#                                                    , np.mean(score_list[:, 2])))


# def exp_on_MoCap(win_size, step, verbose=False):
#     """Experiment on MoCap dataset."""
#     base_path = os.path.join(data_path, 'MoCap/4d/')
#     out_path = os.path.join(output_path, 'MoCap')
#     create_path(out_path)
#     score_list = []
#     params['in_channels'] = 4
#     params['compared_length'] = win_size
#     params['out_channels'] = 4
#     f_list = os.listdir(base_path)
#     f_list.sort()
#     for idx, fname in enumerate(f_list):
#         dataset_path = base_path + fname
#         df = pd.read_csv(dataset_path, sep=' ', usecols=range(0, 4))
#         data = df.to_numpy()
#         groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
#         t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data, win_size, step)

#         prediction = t2s.state_seq
#         prediction = np.array(prediction, dtype=int)
#         result = np.vstack([groundtruth, prediction])
#         np.save(os.path.join(out_path, fname), result)
#         ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
#         score_list.append(np.array([ari, anmi, nmi]))
#         if verbose:
#             print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' % (fname, ari, anmi, nmi))
#     score_list = np.vstack(score_list)
#     print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
#                                                        , np.mean(score_list[:, 1])
#                                                    , np.mean(score_list[:, 2])))


# def exp_on_synthetic(win_size=512, step=100, verbose=False):
#     """Experiment on Synthetic dataset."""

#     out_path = os.path.join(output_path, 'synthetic')
#     create_path(out_path)
#     params['in_channels'] = 4
#     params['compared_length'] = win_size
#     params['out_channels'] = 4
#     prefix = os.path.join(data_path, 'synthetic/test')

#     score_list = []
#     for i in range(100):
#         df = pd.read_csv(prefix + str(i) + '.csv', usecols=range(4), skiprows=1)
#         data = df.to_numpy()
#         df = pd.read_csv(prefix + str(i) + '.csv', usecols=[4], skiprows=1)
#         groundtruth = df.to_numpy(dtype=int).flatten()
#         t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data, win_size, step)
#         prediction = t2s.state_seq
#         prediction = np.array(prediction, dtype=int)
#         result = np.vstack([groundtruth, prediction])
#         np.save(os.path.join(out_path, str(i)), result)
#         ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
#         score_list.append(np.array([ari, anmi, nmi]))
#         if verbose:
#             print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' % (i, ari, anmi, nmi))
#     score_list = np.vstack(score_list)
#     print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
#                                                        , np.mean(score_list[:, 1])
#                                                    , np.mean(score_list[:, 2])))
    
def plot_pdu_time_series(pdu_time, data, category_data):
    n_timepoints, n_features = data.shape
    
    plt.figure(figsize=(20, 12))
    colors = plt.cm.viridis(np.linspace(0, 1, n_features))  # 使用 colormap 为每个特征生成不同的颜色
    # 绘制时间序列数据
    for i in range(n_features):
        plt.plot(pdu_time, data[:, i], label=f'PDU {i+1}', color=colors[i])
    
    plt.xlabel('Time in seconds')
    plt.ylabel('Number of PDUs in 10 Seconds')
    
    # 添加类别数据的透明阴影
    category_colors = {0: 'gray', 1: 'red', 2: 'blue', 3: 'green', 4: 'orange', 5: 'yellow', 6: 'cyan'}  # 定义每个类别的颜色
    unique_categories = np.unique(category_data)
    for category in unique_categories:
        start_idx = None
        for i in range(len(category_data)):
            if category_data[i] == category and start_idx is None:
                start_idx = i
            elif category_data[i] != category and start_idx is not None:
                plt.fill_between(pdu_time[start_idx:i], 0, 1, where=[True]*len(pdu_time[start_idx:i]), color=category_colors[category], alpha=0.3, transform=plt.gca().get_xaxis_transform())
                start_idx = None
        if start_idx is not None:  # 如果最后一段数据也是当前类别
            plt.fill_between(pdu_time[start_idx:], 0, 1, where=[True]*len(pdu_time[start_idx:]), color=category_colors[category], alpha=0.3, transform=plt.gca().get_xaxis_transform())

    # plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    custom_handles = []
    custom_labels = []
    
    for i in range(0, len(labels), 100):
        custom_handles.append(handles[i])
        custom_labels.append(labels[i])
        if i + 100 < len(labels):
            custom_handles.append(plt.Line2D([], [], color='none'))
            custom_labels.append(". . .")
    
    plt.legend(custom_handles, custom_labels, loc='upper right')
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    save_path = "C:\\Ipdu_stastics\\X2E_npy\\pdu2.png"
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()

def exp_on_ActRecTut(win_size, step, verbose=False):
    """Experiment on ActRecTut dataset."""
    out_path = os.path.join(output_path, 'ActRecTut')
    create_path(out_path)
    params['in_channels'] = 10
    params['compared_length'] = win_size
    params['out_channels'] = 4
    score_list = []
    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        for i in range(10):
            dataset_path = os.path.join(data_path, 'ActRecTut/' + dir_name + '/data.mat')
            data = scipy.io.loadmat(dataset_path)
            groundtruth = data['labels'].flatten()
            groundtruth = reorder_label(groundtruth)
            data = data['data'][:, 0:10]
            data = normalize(data)
            t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data, win_size, step)
            prediction = t2s.state_seq + 1
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path, dir_name + str(i)), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' % (dir_name, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
                                                       , np.mean(score_list[:, 1])
                                                   , np.mean(score_list[:, 2])))





def exp_on_PAMAP2(win_size, step, verbose=False):
    """Experiment on PAMAP2 dataset."""
    out_path = os.path.join(output_path, 'PAMAP2')
    create_path(out_path)
    params['in_channels'] = 9
    params['compared_length'] = win_size
    params['out_channels'] = 9

    dataset_path = os.path.join(data_path, 'PAMAP2/Protocol/subject10' + str(1) + '.dat')
    df = pd.read_csv(dataset_path, sep=' ', header=None)
    data = df.to_numpy()
    hand_acc = data[:, 4:7]
    chest_acc = data[:, 21:24]
    ankle_acc = data[:, 38:41]
    data = np.hstack([hand_acc, chest_acc, ankle_acc])
    data = fill_nan(data)
    data = normalize(data)
    t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data, win_size, step)
    score_list = []
    for i in range(1, 9):
        dataset_path = os.path.join(data_path, 'PAMAP2/Protocol/subject10' + str(i) + '.dat')
        df = pd.read_csv(dataset_path, sep=' ', header=None)
        data = df.to_numpy()
        groundtruth = np.array(data[:, 1], dtype=int)
        hand_acc = data[:, 4:7]
        chest_acc = data[:, 21:24]
        ankle_acc = data[:, 38:41]
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
        data = fill_nan(data)
        data = normalize(data)
        t2s.predict(data, win_size, step)
        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' % (i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
                                                       , np.mean(score_list[:, 1])
                                                   , np.mean(score_list[:, 2])))


def exp_on_USC_HAD(win_size, step, verbose=False):
    """Experiment on USC_HAD dataset."""
    out_path = os.path.join(output_path, 'USC-HAD')
    create_path(out_path)
    score_list = []
    score_list2 = []
    f_list = []
    params['in_channels'] = 6
    params['compared_length'] = win_size
    params['kernel_size'] = 3
    params['nb_steps'] = 40

    train, _ = load_USC_HAD(1, 1, data_path)
    train = normalize(train)
    t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(train, win_size, step)
    for subject in range(1, 15):
        for target in range(1, 6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            t2s.predict(data, win_size, step)
            prediction = t2s.state_seq
            t2s.set_clustering_component(DPGMM(13)).predict_without_encode(data, win_size, step)
            prediction2 = t2s.state_seq
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path, 's%d_t%d' % (subject, target)), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            ari2, anmi2, nmi2 = evaluate_clustering(groundtruth, prediction2)
            f1, p, r = evaluate_cut_point(groundtruth, prediction2, 500)
            score_list.append(np.array([ari, anmi, nmi]))
            score_list2.append(np.array([ari2, anmi2, nmi2]))
            f_list.append(np.array([f1, p, r]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' % ('s' + str(subject) + 't' + str(target), ari, anmi, nmi))
                print(
                    'ID: %s, ARI: %f, ANMI: %f, NMI: %f' % ('s' + str(subject) + 't' + str(target), ari2, anmi2, nmi2))
    score_list = np.vstack(score_list)
    score_list2 = np.vstack(score_list2)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
                                                       , np.mean(score_list[:, 1])
                                                   , np.mean(score_list[:, 2])))
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list2[:, 0]) \
                                                       , np.mean(score_list2[:, 1])
                                                   , np.mean(score_list2[:, 2])))


if __name__ == '__main__':
    # time_start = time.time()
    # data_path = os.path.join("C:\\Ipdu_stastics\\X2E_npy\\X2E_297_01096_20240318_130813_20240326_171322_no_someip\\")
    # output_path = os.path.join("C:\\Ipdu_stastics\\X2E_npy\\DP_HSMM")
    # dataset_path = data_path
    # fname = "pdu_data_segment_0.npy"
    # dataset_path = dataset_path + fname
    # X = np.load(dataset_path, allow_pickle=True)
    # data = np.vstack(X)
    # df = pd.DataFrame(data, columns=['Timestamp', 'Signal', 'Quantity1', 'Quantity2', 'Quantity3'])
    # df['Signal_Quantity1'] = df['Signal'] + '_' + df['Quantity1']
    # df_pivot = df.pivot_table(index='Timestamp', columns='Signal_Quantity1', values=['Quantity2'], aggfunc='first')
    # df_pivot = df_pivot.fillna(0)
    # result_array = df_pivot.to_numpy().astype(int)
    # result_array = pca.pca_process(result_array, 18)
    # pdu_time= np.sort(df['Timestamp'].unique())
    # catagory_data = np.load(os.path.join(output_path, fname), allow_pickle=True)
    try:
        exp_on_trace(200, 10, verbose=False) 
    except Exception as e:
        print("An exception occurred: ", str(e))
        traceback.print_exc()
    
        # print('round',i)
        # print('exp_on_synthetic')
        # exp_on_synthetic(256, 50, verbose=False)
        # print('exp_on_MoCap')
        # exp_on_MoCap(256, 50, verbose=False)
        # print('exp_on_ActRecTut')
        # exp_on_ActRecTut(128, 1, verbose=False)
        # print('exp_on_PAMAP2')
        # exp_on_PAMAP2(512, 100, verbose=False)
        # print('exp_on_USC_HAD')
        # exp_on_USC_HAD(512, 50, verbose=False)
        # print('exp_on_UCR_SEG')
        # exp_on_UCR_SEG(512, 50, verbose=False)
    # time_end = time.time()

    # print('time', time_end - time_start)
    # plot_pdu_time_series(pdu_time, result_array, catagory_data)