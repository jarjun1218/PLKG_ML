from models import cnn_encoder_with_rssi, cnn_encoder_without_rssi,cnn_fnn_with_rssi,cnn_fnn_without_rssi,encoder_with_rssi,encoder_without_rssi,fnn_with_rssi,fnn_without_rssi
from dataset import *
#model|dataset


model_data_dict = [
    #[cnn_encoder_with_rssi.cnn_encoder_rssi,csi_rssi_cnn_dataset,"cnn_encoder_rssi"],
    #[cnn_encoder_without_rssi.cnn_encoder,csi_cnn_dataset,"cnn_encoder"],
    #[cnn_fnn_with_rssi.cnn_fnn_rssi,csi_rssi_cnn_dataset,"cnn_fnn_rssi"],
    #[cnn_fnn_without_rssi.cnn_fnn,csi_cnn_dataset,"cnn_fnn"],
    [encoder_with_rssi.encoder_rssi,csi_rssi_dataset,"encoder_rssi"],
    [encoder_without_rssi.encoder,csi_dataset,"encoder"],
    [fnn_with_rssi.fnn_rssi,csi_rssi_dataset,"fnn_rssi"],
    [fnn_without_rssi.fnn,csi_dataset,"fnn"]
]

