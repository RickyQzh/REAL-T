# tse_model.py
import wesep

# Currently not support spex_plus_100, spex_plus_360
def load_tse_model(device, model_name):
    valid_model_names = [
        "bsrnn_vox1",
        "bsrnn_hr_vox1"
    ]
    assert model_name in valid_model_names, f"Invalid model_name: {model_name}. Not in {valid_model_names}"

    if model_name == "bsrnn_vox1":
        print(f"Loading BSRNN model")
        model = wesep.load_model("english")
    elif model_name == "bsrnn_hr_vox1":
        print(f"Loading BSRNN_HR_Vox1 model")
        model = wesep.load_model_local('./wesep/examples/librimix/tse/v2/exp/bsrnn_hr_vox1')
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    model.set_device(device)
    return model

# import wesep
# model_bsrnn = wesep.load_model_local('./examples/librimix/tse/v2/exp/bsrnn_feats_SDR')
# model_tfgridnet = wesep.load_model_local('./examples/librimix/tse/v2/exp/usef_tfgridnet')
if __name__ == "__main__":
    # Example usage
    path = "/root/shared-nvme/open-source/REAL-T/FireRedASR/examples/wav/BAC009S0764W0121.wav"
    model = load_tse_model("cuda", "bsrnn_vox1")
    speech = model.extract_speech(path, path)
    print(speech[0])
    model = load_tse_model("cuda", "bsrnn_hr_vox1")
    speech = model.extract_speech(path, path)
    print(speech[0])
    print("TSE model loaded successfully.")
    # You can now use `model` to perform TSE tasks.