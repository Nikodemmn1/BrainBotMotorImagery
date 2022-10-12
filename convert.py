from Utilities.converters import LargeEEGDataConverter, load_mean_std

conv = LargeEEGDataConverter("./Data/Train", "./Data/EEGLarge/Train")
conv.convert_and_save()

mean_std = load_mean_std("./Data/EEGLarge/Train/Train_mean_std.pkl")
mean = mean_std["mean"]
std = mean_std["std"]

conv = LargeEEGDataConverter("./Data/Test", "./Data/EEGLarge/Test")
conv.convert_and_save(mean, std)

conv = LargeEEGDataConverter("./Data/Val", "./Data/EEGLarge/Val")
conv.convert_and_save(mean, std)
