import json
import keras
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time
from datetime import timedelta
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.layers import Input, MaxPooling2D
from keras.models import Model
from keras.optimizers import *
from keras.utils import np_utils
from mplfinance.original_flavor import volume_overlay
from PIL import Image
from shutil import copyfile, move
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from tqdm import tqdm
from utils.dataset import dataset

matplotlib.use("Agg")


def image2dataset(input, ds_dir, label_file):

    label_dict = {}
    with open(ds_dir + "/" + label_file) as f:
        for line in f:
            key, val = line.split(",")
            label_dict[key] = val.rstrip()

    path = "{}/{}".format(os.getcwd(), input)

    # 이건 왜 하는지 도저히 모르겠네...
    # for filename in os.listdir(path):
    #     # print(filename)
    #     # print(os.getcwd())
    #     if filename != "":
    #         for k, v in label_dict.items():
    #             splitname = filename.split("_")
    #             f, e = os.path.splitext(filename)
    #             # print("[DEBUG] - {}".format(splitname))
    #             # newname = "{}_{}".format(splitname[0], splitname[1])
    #             if f == k:
    #                 # print("{} same with {} with v {}".format(filename, k, v))
    #                 new_name = "{}{}.png".format(v, f)

    #                 os.rename(
    #                     "{}/{}".format(path, filename), "{}/{}".format(path, new_name)
    #                 )
    #                 break

    folders = ["1", "0"]
    for folder in folders:
        if not os.path.exists("{}/classes/{}".format(path, folder)):
            os.makedirs("{}/classes/{}".format(path, folder))

    for filename in os.listdir(path):
        if filename != "" and filename != "classes":
            # print(filename[:1])
            ### 여기에 for k,v in label_dict.items() 돌면서
            f, e = os.path.splitext(filename)
            if label_dict[f] == "1":
                move(
                    "{}/{}".format(path, filename),
                    "{}/classes/1/{}".format(path, filename),
                )
            elif label_dict[f] == "0":
                move(
                    "{}/{}".format(path, filename),
                    "{}/classes/0/{}".format(path, filename),
                )

    print("Done")


# 이게 정답 라벨을 만드는 함수라는데...
def createLabel(csv, ds_dir, seq_len):
    print("Creating label . . .")

    df = pd.read_csv(csv, parse_dates=True, index_col=0)
    df.fillna(0)

    # Date 컬럼을 matplotlib가 이해할 수 있는 부동 소수점 날짜 형식으로...
    df.reset_index(inplace=True)
    df["Date"] = df["Date"].map(mdates.date2num)

    if not os.path.exists(ds_dir):
        os.makedirs(ds_dir)

    filename = csv.split("/")[1]
    symbol = filename.split("_")[0]
    label_file_path = "{}/{}_label_{}.txt".format(ds_dir, filename[:-4], seq_len)
    with open(label_file_path, "w") as the_file:
        for i in range(len(df)):
            # 이것도 폐기된 속성 ix
            # c = df.ix[i : i + int(seq_len), :]
            c = df.iloc[i : i + int(seq_len), :]

            starting = 0
            endvalue = 0
            label = ""

            if len(c) == int(seq_len):
                # starting = c["Close"].iloc[-2]
                starting = c["Open"].iloc[-1]
                endvalue = c["Close"].iloc[-1]
                # print(f'endvalue {endvalue} - starting {starting}')
                tmp_rtn = endvalue / starting - 1
                if tmp_rtn > 0:
                    label = 1
                else:
                    label = 0

                the_file.write("{}-{},{}".format(symbol, i, label))
                the_file.write("\n")
    print("Create label finished.")


def countImage(input):
    num_file = sum([len(files) for r, d, files in os.walk(input)])
    num_dir = sum([len(d) for r, d, files in os.walk(input)])
    print("num of files : {}\nnum of dir : {}".format(num_file, num_dir))


# 캔들차트를 만드는 코드라고...
def ohlc2cs(fname, ds_dir, seq_len, dataset_type, dimension, use_volume):
    print("Converting ohlc to candlestick")
    symbol = fname.split("_")[0]
    symbol = symbol.split("/")[1]
    # print(symbol)
    path = f"{ds_dir}/{seq_len}_{dimension}/{symbol}/{dataset_type}"
    # print(path)
    if not os.path.exists(path):
        os.makedirs(path)

    df = pd.read_csv(fname, parse_dates=True, index_col=0)
    df.fillna(0)
    plt.style.use("dark_background")
    for i in tqdm(range(len(df) - int(seq_len))):
        # ohlc+volume
        c = df.iloc[i : i + int(seq_len), :]
        aa = len(c)
        if len(c) == int(seq_len):
            my_dpi = 96
            fig = plt.figure(
                figsize=(dimension / my_dpi, dimension / my_dpi), dpi=my_dpi
            )
            ax1 = fig.add_subplot(1, 1, 1)
            # 1. Define custom market colors
            mc = mpf.make_marketcolors(
                up="#77d879", down="#db3f3f", inherit=True
            )  # inherit applies these colors to wicks/edges
            # 2. Create a custom style based on those colors
            s = mpf.make_mpf_style(marketcolors=mc)
            # 3. Plot the data using the custom style
            mpf.plot(
                c,
                type="candle",
                ax=ax1,
                scale_width_adjustment=dict(candle=1.5),
                style=s,
            )
            # candlestick2_ochl(
            #     ax1,
            #     c["Open"],
            #     c["Close"],
            #     c["High"],
            #     c["Low"],
            #     width=1,
            #     colorup="#77d879",
            #     colordown="#db3f3f",
            # )
            ax1.grid(False)
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax1.axis("off")

            # create the second axis for the volume bar-plot
            # Add a seconds axis for the volume overlay
            if use_volume:
                ax2 = ax1.twinx()
                # Plot the volume overlay
                bc = volume_overlay(
                    ax2,
                    c["Open"],
                    c["Close"],
                    c["Volume"],
                    colorup="#77d879",
                    colordown="#db3f3f",
                    alpha=0.5,
                    width=1,
                )
                ax2.add_collection(bc)
                ax2.grid(False)
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                ax2.xaxis.set_visible(False)
                ax2.yaxis.set_visible(False)
                ax2.axis("off")
            pngfile = path + "/{}-{}.png".format(symbol, i)
            fig.savefig(pngfile, pad_inches=0, transparent=False)
            plt.close(fig)

            # Alpha 채널 없애기 위한.
            img = Image.open(pngfile)
            img = img.convert("RGB")
            img.save(pngfile)

        # normal length - end

    print("Converting olhc to candlestik finished.")


def preproccess_binclass(command, seq_len, csv, mode, dimension, use_vol, label_file):
    if command == "ohlc2cs":
        ohlc2cs(csv, seq_len, mode, dimension, use_vol)
    if command == "img2dt":
        image2dataset(csv, label_file)
    if command == "countImg":
        countImage(csv)


def run_binary_preprocessing(sym, ds_dir, win_len, dim, mode=None, use_vol=False):
    formatters = {
        "RED": "\033[91m",
        "GREEN": "\033[92m",
        "END": "\033[0m",
    }

    # 이건 나중에 없애고...
    onlytesting = True
    onlytraining = True
    if mode == "test":
        onlytraining = False
    if mode == "train":
        onlytesting = False

    if onlytraining:
        # 훈련 정답지 만들기
        print("{RED}\n훈련 데이터 라벨 생성{END}".format(**formatters))
        # createLabel(f"data/{sym}_training.csv", ds_dir, win_len)
        print("{GREEN}훈련 데이터 라벨 생성 완료\n{END}!".format(**formatters))
    if onlytesting:
        # 시험 정답지 만들기
        print("{RED}\n시험 데이터 라벨 생성{END}".format(**formatters))
        # createLabel(f"data/{sym}_testing.csv", ds_dir, win_len)
        print("{GREEN}시험 데이터 라벨 생성 완료\n{END}".format(**formatters))

    if onlytraining:
        # 훈련 캔들 차트 바꾸기
        print("{RED}\n훈련 데이터 봉 차트 변환{END}".format(**formatters))
        # ohlc2cs(f"data/{sym}_training.csv", ds_dir, win_len, "train", dim, use_vol)
        print("{GREEN}훈련 데이터 봉 차트 변환 완료\n{END}".format(**formatters))
    if onlytesting:
        # 시험 캔들 차트 바꾸기
        print("{RED}\n시험 데이터 봉 차트 변환{END}".format(**formatters))
        ohlc2cs(f"data/{sym}_testing.csv", ds_dir, win_len, "test", dim, use_vol)
        print("{GREEN}시험 데이터 봉 차트 변환 완료\n{END}".format(**formatters))

    if onlytraining:
        # 훈련 데이터 정답 붙이기?
        print("{RED}\nLabelling Training Data{END}".format(**formatters))
        # image2dataset(
        #     f"{ds_dir}/{win_len}_{dim}/{sym}/train",
        #     ds_dir,
        #     f"{sym}_training_label_{win_len}.txt",
        # )
        print("{GREEN}Labelling Training Data Done\n{END}".format(**formatters))
    if onlytesting:
        # 시험 데이터 정답 붙이기?
        print("{RED}\nLabelling Testing Data{END}".format(**formatters))
        image2dataset(
            f"{ds_dir}/{win_len}_{dim}/{sym}/test",
            ds_dir,
            f"{sym}_testing_label_{win_len}.txt",
        )
        print("{GREEN}Labelling Testing Data Done\n{END}".format(**formatters))


def generate_data(pathdir, origindir, targetdir):
    # create folder output
    if not os.path.exists("{}/{}".format(pathdir, targetdir)):
        os.mkdir("{}/{}".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/train".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/test".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/train/0".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train/0".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/train/1".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train/1".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/test/0".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test/0".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/test/1".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test/1".format(pathdir, targetdir))

    counttest = 0
    counttrain = 0
    for root, dirs, files in os.walk("{}/{}".format(pathdir, origindir)):
        for file in files:

            tmp = root.replace("\\", "/")
            tmp_label = tmp.split("/")[-1]

            if tmp_label == "0":
                if "test" in root:
                    origin = "{}/{}".format(root, file)
                    destination = "{}/{}/test/0/{}".format(pathdir, targetdir, file)
                    copyfile(origin, destination)
                    counttest += 1
                elif "train" in root:
                    origin = "{}/{}".format(root, file)
                    destination = "{}/{}/train/0/{}".format(pathdir, targetdir, file)
                    copyfile(origin, destination)
                    counttrain += 1
            elif tmp_label == "1":
                if "test" in root:
                    origin = "{}/{}".format(root, file)
                    destination = "{}/{}/test/1/{}".format(pathdir, targetdir, file)
                    copyfile(origin, destination)
                    counttest += 1
                elif "train" in root:
                    origin = "{}/{}".format(root, file)
                    destination = "{}/{}/train/1/{}".format(pathdir, targetdir, file)
                    copyfile(origin, destination)
                    counttrain += 1

    print(counttest)
    print(counttrain)


def build_dataset(data_directory, img_width):
    # X, y, tags = dataset.dataset(data_directory, int(img_width))
    X, y, tags = dataset(data_directory, int(img_width))
    print(len(tags))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count
    print("train size : {}".format(train_size))
    feature = X
    label = np_utils.to_categorical(y, nb_classes)
    return feature, label, nb_classes


def build_model(SHAPE, nb_classes, bn_axis, seed=None):

    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    # Step 1
    x = Conv2D(
        32,
        3,
        3,
        kernel_initializer="glorot_uniform",
        padding="same",
        activation="relu",
    )(input_layer)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    # Step 1
    x = Conv2D(
        48,
        3,
        3,
        kernel_initializer="glorot_uniform",
        padding="same",
        activation="relu",
    )(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = Dropout(0.25)(x)

    # Step 1
    x = Conv2D(
        64,
        3,
        3,
        kernel_initializer="glorot_uniform",
        padding="same",
        activation="relu",
    )(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    # Step 1
    x = Conv2D(
        96,
        3,
        3,
        kernel_initializer="glorot_uniform",
        padding="same",
        activation="relu",
    )(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = Dropout(0.25)(x)

    # Step 3 - Flattening
    x = Flatten()(x)

    # Step 4 - Full connection

    x = Dense(256, activation="relu")(x)
    # Dropout
    x = Dropout(0.5)(x)

    x = Dense(2, activation="softmax")(x)

    model = Model(input_layer, x)

    return model


# Plot a confusion matrix.
# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("confusion_matrix.png")
    plt.show()


# Plot an ROC. pred - the predictions, y - the expected output.
def plot_roc(pred, y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig("ROC AUC.png")
    plt.show()


def set_tf_env():
    # 경고 메시지 1(Info), 2(Warning) 출력하지 않음
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 0번 GPU만 보이도록 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_deep_cnn(input, epochs, dimension, batch_size, output, channel=3):
    set_tf_env()

    start_time = time.monotonic()
    # dimensions of our images.
    img_width, img_height = dimension, dimension
    SHAPE = (img_width, img_height, channel)
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1

    # channel = 4
    # bn_axis = 4 if K.image_data_format() == 'channels_last' else 1

    data_directory = input

    print("loading dataset")
    X_train, Y_train, nb_classes = build_dataset(
        "{}/train".format(data_directory), dimension
    )
    X_test, Y_test, nb_classes = build_dataset(
        "{}/test".format(data_directory), dimension
    )
    print("number of classes : {}".format(nb_classes))

    model = build_model(SHAPE, nb_classes, bn_axis)

    model.compile(
        optimizer=Adam(lr=1.0e-4), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    # Save Model or creates a HDF5 file
    model.save(
        "{}epochs_{}batch_cnn_model_{}.h5".format(
            epochs, batch_size, data_directory.replace("/", "_")
        ),
        overwrite=True,
    )

    # del model  # deletes the existing model
    predicted = model.predict(X_test)
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    if tp == 0:
        tp = 1
    if tn == 0:
        tn = 1
    if fp == 0:
        fp = 1
    if fn == 0:
        fn = 1
    TPR = float(tp) / (float(tp) + float(fn))
    FPR = float(fp) / (float(fp) + float(tn))
    accuracy = round(
        (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn)), 3
    )
    specitivity = round(float(tn) / (float(tn) + float(fp)), 3)
    sensitivity = round(float(tp) / (float(tp) + float(fn)), 3)
    mcc = round(
        (float(tp) * float(tn) - float(fp) * float(fn))
        / math.sqrt(
            (float(tp) + float(fp))
            * (float(tp) + float(fn))
            * (float(tn) + float(fp))
            * (float(tn) + float(fn))
        ),
        3,
    )

    f_output = open(output, "a")
    f_output.write("=======\n")
    f_output.write("{}epochs_{}batch_cnn\n".format(epochs, batch_size))
    f_output.write("TN: {}\n".format(tn))
    f_output.write("FN: {}\n".format(fn))
    f_output.write("TP: {}\n".format(tp))
    f_output.write("FP: {}\n".format(fp))
    f_output.write("TPR: {}\n".format(TPR))
    f_output.write("FPR: {}\n".format(FPR))
    f_output.write("accuracy: {}\n".format(accuracy))
    f_output.write("specitivity: {}\n".format(specitivity))
    f_output.write("sensitivity : {}\n".format(sensitivity))
    f_output.write("mcc : {}\n".format(mcc))
    f_output.write("{}".format(report))
    f_output.write("=======\n")
    f_output.close()
    end_time = time.monotonic()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))

    plot_roc(y_pred, Y_test)


if __name__ == "__main__":
    ds_dir = "data/cnn_dataset"
    print("binary_preprocessing 작업중...")
    # run_binary_preprocessing("BBNI.JK", ds_dir, 20, 50)

    print("generatedata 작업중...")
    # generate_data("data/cnn_dataset", "20_50/BBNI.JK", "dataset_BBNIJK_20_50")

    print("myDeepCNN 작업중...")
    run_deep_cnn(
        input="data/cnn_dataset/dataset_BBNIJK_20_50",
        epochs=50,
        dimension=50,
        batch_size=8,
        output="outputresult.txt",
    )
