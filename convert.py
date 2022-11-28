from Utilities.converters import LargeEEGDataConverter, BiosemiBDFConverter

# conv = LargeEEGDataConverter("./Data/", "./Data/EEGLarge/")
conv = BiosemiBDFConverter("./DataBDF/", "./DataBDF/Out/")
conv.convert_and_save()
