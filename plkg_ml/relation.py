from models import encoder_with_rssi,encoder_without_rssi,fnn_with_rssi,fnn_without_rssi
from models import cnn_basic,crnet, cnn_speed, cnn_speed_label
from models import cnn_basic_quan, cnn_speed_quan, encoder_basic_quan, fnn_basic_quan
from models import cnn_speed_quan_with_LSTM, cnn_quan_with_LSTM, cnn_with_LSTM
from dataset import *
#model|dataset


model_data_dict = [
    # [encoder_with_rssi.encoder_rssi,csi_rssi_dataset,"encoder_rssi"], #模型|容器|名稱
    [encoder_without_rssi.encoder,csi_dataset,"encoder"],
    # [fnn_with_rssi.fnn_rssi,csi_rssi_dataset,"fnn_rssi"],
    [fnn_without_rssi.fnn,csi_dataset,"fnn"]
]

model_data_cnn_dict = [
    [cnn_basic.cnn_basic,csi_cnn_dataset,"cnn_basic"],
    # [cnn_with_LSTM.cs_net,csi_cnn_lstm_dataset,"cnn_with_LSTM"],
    # [crnet.cr_net,csi_rssi_cnn_dataset,"crnet"]
]

model_data_cnn_speed_dict = [
    [cnn_basic.cnn_basic,csi_cnn_dataset,"cnn_basic"],
    # [cnn_speed.cs_net,csi_cnn_speed_dataset,"cnn_speed"],
]

model_data_quan_dict = [
    [encoder_basic_quan.encoder,csi_quan_dataset,"encoder_basic_quan"],
    [fnn_basic_quan.fnn,csi_quan_dataset,"fnn_basic_quan"],
]

model_data_cnn_quan_dict = [
    [cnn_basic_quan.cnn_basic,csi_cnn_quan_dataset,"cnn_basic_quan"],
    # [cnn_speed_quan_with_LSTM.cs_net,csi_cnn_quan_lstm_dataset,"cnn_quan_with_LSTM"],
    # [cnn_speed_quan.cs_net,csi_cnn_speed_quan_dataset,"cnn_speed_quan"]
]

model_data_cnn_lstm_dict = [
    # [cnn_speed_quan_with_LSTM.cs_net,csi_cnn_quan_lstm_dataset,"cnn_speed_quan_with_LSTM"]
]
