from Utilities.converters import LargeEEGDataConverter

conv = LargeEEGDataConverter("./Data", "./Data/EEGLarge")
conv.convert_and_save()
