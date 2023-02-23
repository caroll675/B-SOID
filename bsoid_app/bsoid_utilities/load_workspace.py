import streamlit as st
import os
import joblib


@st.cache # Cache the function so that it is only computed once
def load_data(path, name):
    with open(os.path.join(path, str.join('', (name, '_data.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]
    # len(data) = 8
    # data[0] = str, path to root directory, e.g. 'C:\\Users\\chang\\DeepLabCut\\main\\JUPYTER\\DLC_Data'
    # data[1] = list, path to subdirectory, e.g. ['/video_copy']
    # data[2] = int, frame rate, e.g. 30
    # data[3] = list, pose_chosen 
    # data[4] = list, prediction csv, e.g.['C:\\Users\\chang\\DeepLabCut\\main\\JUPYTER\\DLC_Data/video_copy\\2022-11-06 13-57-19DLC_resnet50_mainDec6shuffle1_120000.csv']
    # data[5], data[6] = list, data of the prediction csv, aka processed_input_data
    # data[7] = list, sub_threshold 


def query_workspace():
    working_dir = st.sidebar.text_input('Enter the prior B-SOiD working directory:')
    try:
        os.listdir(working_dir)
        st.markdown(
            'You have selected **{}** for prior working directory.'.format(working_dir))
    except FileNotFoundError:
        st.error('No such directory')
    files = [i for i in os.listdir(working_dir) if os.path.isfile(os.path.join(working_dir, i)) and \
             '_data.sav' in i and not '_accuracy' in i and not '_coherence' in i]
    bsoid_variables = [files[i].partition('_data.sav')[0] for i in range(len(files))]
    bsoid_prefix = []
    for var in bsoid_variables:
        if var not in bsoid_prefix:
            bsoid_prefix.append(var)
    prefix = st.selectbox('Select prior B-SOiD prefix', bsoid_prefix)
    try:
        st.markdown('You have selected **{}_XXX.sav** for prior prefix.'.format(prefix))
    except TypeError:
        st.error('Please input a prior prefix to load workspace.')
    return working_dir, prefix


@st.cache
def load_feats(path, name):
    with open(os.path.join(path, str.join('', (name, '_feats.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data] 


@st.cache
def load_embeddings(path, name):
    with open(os.path.join(path, str.join('', (name, '_embeddings.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]
# len(data) = 2
# data[0] = np.array, sampled_features, e.g.sampled_features.shape = (47999, 49)
# data[1] = np.array, sampled_embeddings, e.g.sampled_embeddings.shape = (47999, 16)

@st.cache
def load_clusters(path, name):
    with open(os.path.join(path, str.join('', (name, '_clusters.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]
# len(data) = 4
# data[0] = np.array, unknown 
# data[1] = np.arrary, assignments
# data[2] = np.array, assign_prob
# data[3] = np.array, soft_assignments

@st.cache(allow_output_mutation=True)
def load_classifier(path, name):
    with open(os.path.join(path, str.join('', (name, '_randomforest.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


@st.cache
def load_predictions(path, name):
    with open(os.path.join(path, str.join('', (name, '_predictions.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


def load_new_feats(path, name):
    with open(os.path.join(path, str.join('', (name, '_new_feats.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]




