import os
import numpy as np
import librosa
import soundfile as sf

DATASET_PATH = r"C:\Users\armag\Desktop\NEWDATASET"
NEW_DATASET_PATH = r"C:\Users\armag\Desktop\Augmant"
SAMPLE_RATE = 22050
NOISE_FACTOR = 0.5
SHIFT_DIRECTION = "right"
PITCH_FOCTOR = 1.5
SPEED_FACTOR = 0.8
SHIFT_MAX = 0.1

def manilate(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shifting(data, shift_max, shift_direction, sampling_rate=1024):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == "right":
        shift = -shift
    elif shift_direction == "both":
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def changing_pitch(data, pitch_factor, sampling_rate = 1024):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def changing_speed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)

def creating_new_file(signal, file_name, new_dateset_path):

    if not os.path.exists(new_dateset_path):
        os.makedirs(new_dateset_path)

    new_file_name = file_name.split(".")[0] + str("_augmented")
    new_file_name = new_dateset_path + str("\\") + new_file_name + str(".wav")
    sf.write(new_file_name, signal, 22050, 'PCM_24')
    print("Data is created: " + new_file_name)


def unvoiced_to_voiced_signals(datapath):

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(datapath)):

        if dirpath is not datapath:
            labels = dirpath.split("\\")[-1]
            data_set_file = NEW_DATASET_PATH + str("\\") + str(labels)

            for file in filenames:

                try:
                    file_path = os.path.join(dirpath, file)
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                    signal = np.array(signal)
                    new_sig = manilate(signal, noise_factor=NOISE_FACTOR)
                    new_sig = changing_pitch(new_sig, pitch_factor=PITCH_FOCTOR, sampling_rate=1024)
                    new_sig = changing_speed(new_sig, speed_factor=SPEED_FACTOR)

                    new_sig = np.array(new_sig, dtype=float)

                    creating_new_file(new_sig, file, data_set_file)
                except:
                    print(file, " is passed")
if __name__=="__main__":
    unvoiced_to_voiced_signals(DATASET_PATH)