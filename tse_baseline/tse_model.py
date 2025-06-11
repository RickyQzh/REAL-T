# tse_model.py
import wesep

# Currently not support spex_plus_100, spex_plus_360
def load_tse_model(device, model_name):
    valid_model_names = [
        "bsrnn_vox1",
        "bsrnn_100",
        "bsrnn_360",
        "bsrnn_hr_100",
        "bsrnn_hr_SDR_360",
        "bsrnn_hr_vox1",
        "bsrnn_hr_360",
        "tfgridnet_100",
        "spex_plus_100",
        "spex_plus_360"
    ]
    assert model_name in valid_model_names, f"Invalid model_name: {model_name}. Not in {valid_model_names}"

    if model_name == "bsrnn_vox1":
        print(f"Loading BSRNN model")
        model = wesep.load_model("english")

    elif model_name == "bsrnn_100":
        print(f"Loading BSRNN_100 model")
        model = wesep.load_model_local('./wesep/examples/librimix/tse/v2/exp/bsrnn_100')
    
    elif model_name == "bsrnn_360":
        print(f"Loading BSRNN_360 model")
        model = wesep.load_model_local('./wesep/examples/librimix/tse/v2/exp/bsrnn_360')
    
    elif model_name == "bsrnn_hr_vox1":
        print(f"Loading BSRNN_HR_Vox1 model")
        model = wesep.load_model_local('./wesep/examples/librimix/tse/v2/exp/bsrnn_hr_vox1')    

    elif model_name == "bsrnn_hr_100":
        print(f"Loading BSRNN_HR_100 model")
        model = wesep.load_model_local('./wesep/examples/librimix/tse/v2/exp/bsrnn_hr_100')

    elif model_name == "bsrnn_hr_360":
        print(f"Loading BSRNN_HR_360 model")
        model = wesep.load_model_local('./wesep/examples/librimix/tse/v2/exp/bsrnn_hr_360')

    elif model_name == "bsrnn_hr_SDR_360":
        print(f"Loading BSRNN_HR model")
        model = wesep.load_model_local('./wesep/examples/librimix/tse/v2/exp/bsrnn_feats_SDR')

    elif model_name == "tfgridnet_100":
        print(f"Loading TF-GridNet_100 model")
        model = wesep.load_model_local('./wesep/examples/librimix/tse/v2/exp/usef_tfgridnet_100')

    elif model_name == "spex_plus_100":
        print(f"Loading Spex+_100 model")
        model = wesep.load_model_local('./wesep/examples/librimix/tse/v2/exp/spex_plus_100')
    
    elif model_name == "spex_plus_360":
        print(f"Loading Spex+_360 model")
        model = wesep.load_model_local('./wesep/examples/librimix/tse/v2/exp/spex_plus_360')

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    model.set_device(device)
    return model

# # import wesep
# # model_bsrnn = wesep.load_model_local('./examples/librimix/tse/v2/exp/bsrnn_feats_SDR')
# # model_tfgridnet = wesep.load_model_local('./examples/librimix/tse/v2/exp/usef_tfgridnet')
# if __name__ == "__main__":
#     # Example usage
#     path = "/root/shared-nvme/open-source/REAL-T/FireRedASR/examples/wav/BAC009S0764W0121.wav"
#     model = load_tse_model("cuda", "bsrnn_vox1")
#     speech = model.extract_speech(path, path)
#     print(speech[0])
#     model = load_tse_model("cuda", "bsrnn_hr_vox1")
#     speech = model.extract_speech(path, path)
#     print(speech[0])
#     print("TSE model loaded successfully.")
#     # You can now use `model` to perform TSE tasks.