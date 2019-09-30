import keras
from keras.layers import Input, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPool2D, Dropout, Dense, LSTM, GRU
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History 
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from keras import models
from keras.models import Model
from imgaug import augmenters
from random import randint
import pandas as pd
import numpy as np
import cv2
import time
import glob, os 
from skimage import io, transform
import json
import datetime
from tqdm import tqdm_notebook
import re
import hw2

def get_features_by_days(df, days):
    days.sort()
    _tmp_df = df.copy(deep=True)
    for num_days_back in days:
        _tmp = [None] * num_days_back
        _tmp.extend(df.iloc[:-num_days_back, :]["DailyAdmission"].values.tolist())
        _tmp_df["{}DaysBack".format(num_days_back)] = _tmp
    _tmp_df = _tmp_df.dropna()
    cols = _tmp_df.columns.tolist()
    return _tmp_df[cols[1:] + [cols[0]]]

def set_weeks_after_peak(df):
    _tmp_df = df.copy(deep=True)
    _tmp_df["WeeksAfterPeak"] = np.nan
    for season_number in range(0,5):
        first_index = _tmp_df.loc[_tmp_df["Season"]==season_number].index.tolist()[0]
        week_breaks = list(range(0, _tmp_df.loc[df["Season"]==season_number].shape[0], 7))
        for ii in range(1, len(week_breaks)):
            _tmp_df.loc[week_breaks[ii-1]+first_index:week_breaks[ii]+first_index, ["WeeksAfterPeak"]] = week_breaks[ii-1]/7
        if season_number == 4:
            _tmp_df.loc[week_breaks[-1] + first_index:_tmp_df.loc[df["Season"]==season_number].shape[0] + first_index, ["WeeksAfterPeak"]] = 11
        else:
            _tmp_df.loc[week_breaks[-1] + first_index:_tmp_df.loc[df["Season"]==season_number].shape[0] + first_index, ["WeeksAfterPeak"]] = 9
    return _tmp_df

def normalize_df(df, sc):
    pd.options.mode.chained_assignment = None 
    _tmp_df = df.copy(deep=True)
    _tmp_df.loc[:, ["DailyAdmission"]] = sc.transform(_tmp_df["DailyAdmission"].values.reshape(-1,1)).flatten()
    for column in [column for column in train_df.columns.tolist() if "DaysBack" in column]:
        _tmp_df.loc[:, [column]] = sc.transform(_tmp_df[column].values.reshape(-1,1)).flatten()
    _tmp_df["Month"] = _tmp_df["Month"]/12
    _tmp_df["Day"] = _tmp_df["Day"]/12
    _tmp_df["Year"] = _tmp_df["Year"]/2018
    _tmp_df["DayOfWeek"] = _tmp_df["DayOfWeek"]/6
    _tmp_df["WeekNumber"] = _tmp_df["WeekNumber"]/52
    _tmp_df["Season"] = _tmp_df["Season"]/4
    _tmp_df["WeeksAfterPeak"] = _tmp_df["WeeksAfterPeak"]/11
    return _tmp_df

def create_sequence_df(df, look_back=5, foresight=7):
    X, Y = [], []
    _tmp_df = df.copy(deep=True)
    for i in range(df.shape[0]-look_back-foresight):
        X.append(df.iloc[i:(i+look_back),:].values)
        Y.append(df.iloc[i+look_back+foresight]["DailyAdmission"])
    return np.array(X), np.array(Y)

def process_csv(file_name, date_column, sc):
    # read data
    df = pd.read_csv(
        filepath_or_buffer=file_name
    )
    df["Month"] = df[date_column].astype(str).str.split("/").str[0].astype(int)
    df["Day"] = df[date_column].astype(str).str.split("/").str[1].astype(int)
    df["Year"] = df[date_column].astype(str).str.split("/").str[2].astype(int)
    dates = df[date_column].tolist()
    df[date_column] = pd.to_datetime(df[date_column])
    df["DayOfWeek"] = df[date_column].dt.dayofweek
    df["WeekNumber"] = df[date_column].dt.week
    df["Season"] = np.nan
    df.loc[
        (df["WeekNumber"]>43) | 
        (df["WeekNumber"]<3)  | 
        ((df["WeekNumber"]<=3) & (df["DayOfWeek"]<6)) | 
        ((df["WeekNumber"]==43) & (df["DayOfWeek"]==6)),
    ["Season"]] = 4
    df.loc[
        ((df["WeekNumber"]>3) & (df["WeekNumber"]<13)) | 
        ((df["WeekNumber"]==3) & (df["DayOfWeek"]==6)) |
        ((df["WeekNumber"]==13) & (df["DayOfWeek"]<6))
    ,["Season"]] = 0
    df.loc[
        ((df["WeekNumber"]>13) & (df["WeekNumber"]<23)) | 
        ((df["WeekNumber"]==13) & (df["DayOfWeek"]==6)) |
        ((df["WeekNumber"]==23) & (df["DayOfWeek"]<6))
    ,["Season"]] = 1
    df.loc[
        ((df["WeekNumber"]>23) & (df["WeekNumber"]<33)) | 
        ((df["WeekNumber"]==23) & (df["DayOfWeek"]==6)) |
        ((df["WeekNumber"]==33) & (df["DayOfWeek"]<6))
    ,["Season"]] = 2
    df.loc[
        ((df["WeekNumber"]>33) & (df["WeekNumber"]<43)) | 
        ((df["WeekNumber"]==33) & (df["DayOfWeek"]==6)) |
        ((df["WeekNumber"]==43) & (df["DayOfWeek"]<6))
    ,["Season"]] = 3
    df = get_features_by_days(df, list(range(1, 8)))
    df = set_weeks_after_peak(df)
    norm_df = normalize_df(df, sc)
    return(create_sequence_df(norm_df))

def run_lstm(
    norm_train_x,
    norm_train_y,
    norm_val_x,
    norm_val_y,
    dropout,
    recurrent_dropout,
    batch_size,
    filter_size,
    epochs = 60,
    verbose = 0
):
    output_path = "./model_output/"
    modelname = "LSTM_numEpoch-{}_dropout-{}_recurrentDropout-{}_batchSize-{}_filterSize-{}".format(
        epochs,
        dropout, 
        recurrent_dropout,
        batch_size,
        filter_size
    )
    with open("lstm.json", "r") as json_file:
        model_dict = json.load(json_file)
    if modelname in model_dict.keys():
        return
    print(modelname)
    filepath = output_path + modelname + "_epoch-{epoch:02d}_loss-{loss:.4f}.hdf5"
    input_layer = Input(shape=(norm_train_x.shape[1], norm_train_x.shape[2]))
    lstm_layer = LSTM(
        filter_size,
        dropout=dropout, 
        recurrent_dropout=recurrent_dropout
    )(input_layer)
    output_layer = Dense(
        1, 
        activation="linear"
    )(lstm_layer)
    model = Model(input_layer, output_layer)
    model.compile(
        loss="mae", 
        optimizer="adam" ,
        metrics=["mean_absolute_error"]
    )
    history = model.fit(
        norm_train_x, 
        norm_train_y,
        validation_data=(
            norm_val_x,
            norm_val_y
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[
            ModelCheckpoint(
                filepath,
                monitor="loss",
                verbose=verbose,
                save_best_only=True,
                mode="min"
            )
        ]
    )
    model_dict[modelname] = history.history
    with open("lstm.json", 'w') as outfile:
        json.dump(model_dict, outfile)

def compare_lstm_filtersize(
        batch_size,
        max_epoch
):
    with open("lstm.json", "r") as json_file:
        model_dict = json.load(json_file)
    models = [
        re.findall(
            r"(GRU|LSTM)_numEpoch-([0-9]+)_dropout-([0-9]+\.[0-9]+)"
            + r"_recurrentDropout-([0-9]+\.[0-9]+)_batchSize-([0-9]+)"
            + r"_filterSize-([0-9]+)", 
            model
        )[0]
        for model in model_dict.keys()]
    model_df = pd.DataFrame(
        {
            "Type":[model[0] for model in models],
            "Epochs":[int(model[1]) for model in models],
            "Dropout":[float(model[2]) for model in models],
            "RecurrentDropout":[float(model[3]) for model in models],
            "BatchSize":[int(model[4]) for model in models],
            "FilterSize":[int(model[5]) for model in models],
            "Name":list(model_dict.keys())
        }
    )
    model_df = model_names = model_df.loc[
        (model_df["BatchSize"] == batch_size) &
        (model_df["Dropout"] == 0.0) &
        (model_df["RecurrentDropout"] == 0.0)
    ]
    filter_sizes = model_df["FilterSize"].tolist()
    model_losses = [
        (model_dict[model]["loss"], model_dict[model]["val_loss"])
        for model in model_df["Name"].tolist()
    ]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5))
    fig.tight_layout()
    for ii in range(1, len(filter_sizes)+1):
        model_loss, model_val_loss = model_losses[ii-1]
        a0.plot(model_loss[0:max_epoch+1], alpha=ii * 1/len(filter_sizes),
                c="red", label="Filter Size: {}".format(filter_sizes[ii-1]))
        a1.plot(model_val_loss[0:max_epoch+1], alpha=ii * 1/len(filter_sizes),
                c="green", label="Filter Size: {}".format(filter_sizes[ii-1]))
    a0.set_title("Train Loss", fontsize=20)
    a1.set_title("Validation Loss", fontsize=20)
    for ax in [a0, a1]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel("Epochs", fontsize=15)
        ax.set_ylabel("Loss", fontsize=15)
        ax.legend()
    plt.show()

def compare_lstm_batchSize(
        filter_size,
        max_epoch
):
    with open("lstm.json", "r") as json_file:
        model_dict = json.load(json_file)
    models = [
        re.findall(
            r"(GRU|LSTM)_numEpoch-([0-9]+)_dropout-([0-9]+\.[0-9]+)"
            + r"_recurrentDropout-([0-9]+\.[0-9]+)_batchSize-([0-9]+)"
            + r"_filterSize-([0-9]+)", 
            model
        )[0]
        for model in model_dict.keys()]
    model_df = pd.DataFrame(
        {
            "Type": [model[0] for model in models],
            "Epochs": [int(model[1]) for model in models],
            "Dropout": [float(model[2]) for model in models],
            "RecurrentDropout": [float(model[3]) for model in models],
            "BatchSize": [int(model[4]) for model in models],
            "FilterSize": [int(model[5]) for model in models],
            "Name": list(model_dict.keys())
        }
    )
    model_df = model_names = model_df.loc[
        (model_df["FilterSize"] == filter_size) &
        (model_df["Dropout"] == 0.0) &
        (model_df["RecurrentDropout"] == 0.0)
    ]
    batch_sizes = model_df["BatchSize"].tolist()
    model_losses = [(model_dict[model]["loss"], model_dict[model]["val_loss"])
                    for model in model_df["Name"].tolist()]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5))
    fig.tight_layout()
    for ii in range(1, len(batch_sizes)+1):
        model_loss, model_val_loss = model_losses[ii-1]
        a0.plot(model_loss[0:max_epoch+1], alpha=ii * 1/len(batch_sizes),
                c="red", label="Batch Size: {}".format(batch_sizes[ii-1]))
        a1.plot(model_val_loss[0:max_epoch+1], alpha=ii * 1/len(batch_sizes),
                c="green", label="Batch Size: {}".format(batch_sizes[ii-1]))
    a0.set_title("Train Loss", fontsize=20)
    a1.set_title("Validation Loss", fontsize=20)
    for ax in [a0, a1]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel("Epochs", fontsize=15)
        ax.set_ylabel("Loss", fontsize=15)
        ax.legend()
    plt.show()

def plot_lstm(
    sc,
    norm_all_x,
    norm_all_y,
    dates,
    batch_size,
    filter_size,
    epoch,
    dropout,
    recurrent_dropout
):
    if epoch > 9:
        modelname = "LSTM_numEpoch-60_dropout-{}_recurrentDropout-{}_batchSize-{}_filterSize-{}_epoch-{}".format(
            dropout/100, 
            recurrent_dropout/100,
            batch_size,
            filter_size,
            epoch
        )
    else:
        modelname = "LSTM_numEpoch-60_dropout-{}_recurrentDropout-{}_batchSize-{}_filterSize-{}_epoch-0{}".format(
            dropout/100, 
            recurrent_dropout/100,
            batch_size,
            filter_size,
            epoch
        )
    model_files = [model_file for model_file in os.listdir("model_output") if modelname in model_file]
    if len(model_files) != 1:
        print("Could not find specific model")
    with open("lstm.json", "r") as json_file:
        model_dict = json.load(json_file)
    model_file = model_files[0]
    model = load_model("model_output/" + model_file)
    predictions = sc.inverse_transform(model.predict(norm_all_x))[6:]
    actual = sc.inverse_transform(np.reshape(norm_all_y, (-1, 1)))[:-6]
    resid = np.remainder(predictions, actual)
    fig, axs = plt.subplots(
        2, 
        1,
        figsize=(10, 10)
    )
    axs[0].plot(actual, label="Actual", color="black")
    axs[0].plot(predictions, label="Predicted", color="green")

    axs[0].set_ylabel('Number of Admissions', fontsize=16)
    axs[0].set_xlabel('Date', fontsize=16)   
    fig.canvas.draw()
    labels = [item.get_text() for item in axs[0].get_xticklabels()]
    _tmp = [labels[0]]
    labels = labels[1:]
    _tmp.extend([dates[int(label)] for label in labels if not label[0] == "−" and int(label) < len(dates)])
    axs[0].set_xticklabels(_tmp, rotation=0)
    train_loss = model_dict["_".join(modelname.split("_")[:-1])]["loss"]
    val_loss = model_dict["_".join(modelname.split("_")[:-1])]["val_loss"]
    axs[1].set_ylabel('Loss', fontsize=16)
    axs[1].set_xlabel('Epoch', fontsize=16)
    axs[1].plot(train_loss[0:epoch+2], c="red", label="Train Loss")
    axs[1].plot(val_loss[0:epoch+2], c="green", label="Validation Loss")
    axs[1].scatter(epoch, train_loss[epoch], c="black")
    axs[1].scatter(epoch, val_loss[epoch], c="black")
    axs[1].axvline(x=epoch, c="black")
    axs[1].legend()
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    plt.show()

def compare_gru_filtersize(
        batch_size,
        max_epoch
):
    with open("gru.json", "r") as json_file:
        model_dict = json.load(json_file)
    models = [
        re.findall(
            r"(GRU|LSTM)_numEpoch-([0-9]+)_dropout-([0-9]+\.[0-9]+)"
            + r"_recurrentDropout-([0-9]+\.[0-9]+)_batchSize-([0-9]+)"
            + r"_filterSize-([0-9]+)", 
            model
        )[0]
        for model in model_dict.keys()]
    model_df = pd.DataFrame(
        {
            "Type":[model[0] for model in models],
            "Epochs":[int(model[1]) for model in models],
            "Dropout":[float(model[2]) for model in models],
            "RecurrentDropout":[float(model[3]) for model in models],
            "BatchSize":[int(model[4]) for model in models],
            "FilterSize":[int(model[5]) for model in models],
            "Name":list(model_dict.keys())
        }
    )
    model_df = model_names = model_df.loc[
        (model_df["BatchSize"] == batch_size) &
        (model_df["Dropout"] == 0.0) &
        (model_df["RecurrentDropout"] == 0.0)
    ]
    filter_sizes = model_df["FilterSize"].tolist()
    model_losses = [
        (model_dict[model]["loss"], model_dict[model]["val_loss"])
        for model in model_df["Name"].tolist()
    ]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5))
    fig.tight_layout()
    for ii in range(1, len(filter_sizes)+1):
        model_loss, model_val_loss = model_losses[ii-1]
        a0.plot(model_loss[0:max_epoch+1], alpha=ii * 1/len(filter_sizes),
                c="red", label="Filter Size: {}".format(filter_sizes[ii-1]))
        a1.plot(model_val_loss[0:max_epoch+1], alpha=ii * 1/len(filter_sizes),
                c="green", label="Filter Size: {}".format(filter_sizes[ii-1]))
    a0.set_title("Train Loss", fontsize=20)
    a1.set_title("Validation Loss", fontsize=20)
    for ax in [a0, a1]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel("Epochs", fontsize=15)
        ax.set_ylabel("Loss", fontsize=15)
        ax.legend()
    plt.show()


def compare_gru_batchSize(
        filter_size,
        max_epoch
):
    with open("gru.json", "r") as json_file:
        model_dict = json.load(json_file)
    models = [
        re.findall(
            r"(GRU|LSTM)_numEpoch-([0-9]+)_dropout-([0-9]+\.[0-9]+)"
            + r"_recurrentDropout-([0-9]+\.[0-9]+)_batchSize-([0-9]+)"
            + r"_filterSize-([0-9]+)", 
            model
        )[0]
        for model in model_dict.keys()]
    model_df = pd.DataFrame(
        {
            "Type": [model[0] for model in models],
            "Epochs": [int(model[1]) for model in models],
            "Dropout": [float(model[2]) for model in models],
            "RecurrentDropout": [float(model[3]) for model in models],
            "BatchSize": [int(model[4]) for model in models],
            "FilterSize": [int(model[5]) for model in models],
            "Name": list(model_dict.keys())
        }
    )
    model_df = model_names = model_df.loc[
        (model_df["FilterSize"] == filter_size) &
        (model_df["Dropout"] == 0.0) &
        (model_df["RecurrentDropout"] == 0.0)
    ]
    batch_sizes = model_df["BatchSize"].tolist()
    model_losses = [(model_dict[model]["loss"], model_dict[model]["val_loss"])
                    for model in model_df["Name"].tolist()]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5))
    fig.tight_layout()
    for ii in range(1, len(batch_sizes)+1):
        model_loss, model_val_loss = model_losses[ii-1]
        a0.plot(model_loss[0:max_epoch+1], alpha=ii * 1/len(batch_sizes),
                c="red", label="Batch Size: {}".format(batch_sizes[ii-1]))
        a1.plot(model_val_loss[0:max_epoch+1], alpha=ii * 1/len(batch_sizes),
                c="green", label="Batch Size: {}".format(batch_sizes[ii-1]))
    a0.set_title("Train Loss", fontsize=20)
    a1.set_title("Validation Loss", fontsize=20)
    for ax in [a0, a1]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel("Epochs", fontsize=15)
        ax.set_ylabel("Loss", fontsize=15)
        ax.legend()
    plt.show()

def plot_gru(
    sc,
    norm_all_x,
    norm_all_y,
    dates,
    batch_size,
    filter_size,
    epoch,
    dropout,
    recurrent_dropout
):
    if epoch > 9:
        modelname = "GRU_numEpoch-60_dropout-{}_recurrentDropout-{}_batchSize-{}_filterSize-{}_epoch-{}".format(
            dropout/100, 
            recurrent_dropout/100,
            batch_size,
            filter_size,
            epoch
        )
    else:
        modelname = "GRU_numEpoch-60_dropout-{}_recurrentDropout-{}_batchSize-{}_filterSize-{}_epoch-0{}".format(
            dropout/100, 
            recurrent_dropout/100,
            batch_size,
            filter_size,
            epoch
        )
    model_files = [model_file for model_file in os.listdir("model_output") if modelname in model_file]
    if len(model_files) != 1:
        print("Could not find specific model")
    with open("gru.json", "r") as json_file:
        model_dict = json.load(json_file)
    model_file = model_files[0]
    model = load_model("model_output/" + model_file)
    predictions = sc.inverse_transform(model.predict(norm_all_x))[6:]
    actual = sc.inverse_transform(np.reshape(norm_all_y, (-1, 1)))[:-6]
    resid = np.remainder(predictions, actual)
    fig, axs = plt.subplots(
        2, 
        1,
        figsize=(10, 10)
    )
    axs[0].plot(actual, label="Actual", color="black")
    axs[0].plot(predictions, label="Predicted", color="green")

    axs[0].set_ylabel('Number of Admissions', fontsize=16)
    axs[0].set_xlabel('Date', fontsize=16)   
    fig.canvas.draw()
    labels = [item.get_text() for item in axs[0].get_xticklabels()]
    _tmp = [labels[0]]
    labels = labels[1:]
    _tmp.extend([dates[int(label)] for label in labels if not label[0] == "−" and int(label) < len(dates)])
    axs[0].set_xticklabels(_tmp, rotation=0)
    train_loss = model_dict["_".join(modelname.split("_")[:-1])]["loss"]
    val_loss = model_dict["_".join(modelname.split("_")[:-1])]["val_loss"]
    axs[1].set_ylabel('Loss', fontsize=16)
    axs[1].set_xlabel('Epoch', fontsize=16)
    axs[1].plot(train_loss[0:epoch+2], c="red", label="Train Loss")
    axs[1].plot(val_loss[0:epoch+2], c="green", label="Validation Loss")
    axs[1].scatter(epoch, train_loss[epoch], c="black")
    axs[1].scatter(epoch, val_loss[epoch], c="black")
    axs[1].axvline(x=epoch, c="black")
    axs[1].legend()
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    plt.show()


def compare_conv_kernelSize(
    batch_size,
    kernel_size,
    max_epoch,
):
    with open("conv1d.json", "r") as json_file:
        model_dict = json.load(json_file)
    models = [
        re.findall(
            r"(CONV1D)_numEpoch-([0-9]+)_kernelSize-([0-9]+)"
            + r"_batchSize-([0-9]+)"
            + r"_filterSize-([0-9]+)", 
            model
        )[0]
        for model in model_dict.keys()]
    model_df = pd.DataFrame(
        {
            "Type":[model[0] for model in models],
            "Epochs":[int(model[1]) for model in models],
            "KernelSize":[int(model[2]) for model in models],
            "BatchSize":[int(model[3]) for model in models],
            "FilterSize":[int(model[4]) for model in models],
            "Name":list(model_dict.keys())
        }
    )
    model_df = model_names = model_df.loc[
        (model_df["BatchSize"] == batch_size)  & 
        (model_df["KernelSize"] == kernel_size)
    ]
    filter_sizes = model_df["FilterSize"].tolist()
    model_losses = [
        (model_dict[model]["loss"], model_dict[model]["val_loss"])
        for model in model_df["Name"].tolist()
    ]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5))
    fig.tight_layout()
    for ii in range(1, len(filter_sizes)+1):
        model_loss, model_val_loss = model_losses[ii-1]
        a0.plot(model_loss[0:max_epoch+1], alpha=ii * 1/len(filter_sizes),
                c="red", label="Filter Size: {}".format(filter_sizes[ii-1]))
        a1.plot(model_val_loss[0:max_epoch+1], alpha=ii * 1/len(filter_sizes),
                c="green", label="Filter Size: {}".format(filter_sizes[ii-1]))
    a0.set_title("Train Loss", fontsize=20)
    a1.set_title("Validation Loss", fontsize=20)
    for ax in [a0, a1]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel("Epochs", fontsize=15)
        ax.set_ylabel("Loss", fontsize=15)
        ax.legend()
    plt.show()

def plot_conv(
    sc,
    norm_all_x,
    norm_all_y,
    dates,
    kernel_size,
    batch_size,
    filter_size,
    epoch
):
    if epoch > 9:
        modelname = "CONV1D_numEpoch-60_kernelSize-{}_batchSize-{}_filterSize-{}_epoch-{}".format(
            kernel_size,
            batch_size,
            filter_size,
            epoch
        )
    else:
        modelname = "CONV1D_numEpoch-60_kernelSize-{}_batchSize-{}_filterSize-{}_epoch-0{}".format(
            kernel_size,
            batch_size,
            filter_size,
            epoch
        )
    print(modelname)
    model_files = [model_file for model_file in os.listdir("model_output") if modelname in model_file]
    if len(model_files) != 1:
        print("Could not find specific model")
    with open("conv1d.json", "r") as json_file:
        model_dict = json.load(json_file)
    model_file = model_files[0]
    model = load_model("model_output/" + model_file)
    predictions = sc.inverse_transform(model.predict(norm_all_x))[6:]
    actual = sc.inverse_transform(np.reshape(norm_all_y, (-1, 1)))[:-6]
    resid = np.remainder(predictions, actual)
    fig, axs = plt.subplots(
        2, 
        1,
        figsize=(10, 10)
    )
    axs[0].plot(actual, label="Actual", color="black")
    axs[0].plot(predictions, label="Predicted", color="green")
    axs[0].set_ylabel('Number of Admissions', fontsize=16)
    axs[0].set_xlabel('Date', fontsize=16)   
    fig.canvas.draw()
    labels = [item.get_text() for item in axs[0].get_xticklabels()]
    _tmp = [labels[0]]
    labels = labels[1:]
    _tmp.extend([dates[int(label)] for label in labels if not label[0] == "−" and int(label) < len(dates)])
    axs[0].set_xticklabels(_tmp, rotation=0)
    train_loss = model_dict["_".join(modelname.split("_")[:-1])]["loss"]
    val_loss = model_dict["_".join(modelname.split("_")[:-1])]["val_loss"]
    axs[1].set_ylabel('Loss', fontsize=16)
    axs[1].set_xlabel('Epoch', fontsize=16)
    axs[1].plot(train_loss[0:epoch+2], c="red", label="Train Loss")
    axs[1].plot(val_loss[0:epoch+2], c="green", label="Validation Loss")
    axs[1].scatter(epoch, train_loss[epoch], c="black")
    axs[1].scatter(epoch, val_loss[epoch], c="black")
    axs[1].axvline(x=epoch, c="black")
    axs[1].legend()
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    plt.show()






