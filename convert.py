from Utilities.converters import LargeEEGDataConverter, BiosemiBDFConverter
from Utilities.merger import Merger

#conv = LargeEEGDataConverter("./Data/", "./Data/EEGLarge/")
#conv.convert_and_save()

conv = BiosemiBDFConverter("./DataBDF/", "./DataBDF/DataTest/")
conv.convert_and_save()

#merger = Merger("./DataBDF/Out/", "./Data/EEGLarge/", "./DataMerged/")
#merger.merge()
