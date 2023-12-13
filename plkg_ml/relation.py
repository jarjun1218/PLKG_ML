from models import encoder_with_rssi,encoder_without_rssi,fnn_with_rssi,fnn_without_rssi
from models import cnn_basic,crnet
from dataset import *
#model|dataset


model_data_dict = [
    [encoder_with_rssi.encoder_rssi,csi_rssi_dataset,"encoder_rssi"],
    [encoder_without_rssi.encoder,csi_dataset,"encoder"],
    [fnn_with_rssi.fnn_rssi,csi_rssi_dataset,"fnn_rssi"],
    [fnn_without_rssi.fnn,csi_dataset,"fnn"]
]

model_data_cnn_dict = [
    [cnn_basic.cnn_basic,csi_cnn_dataset,"cnn_basic"],
    [crnet.cr_net,csi_rssi_cnn_dataset,"crnet"]
]

