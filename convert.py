from Utilities.converters import LargeEEGDataConverter, BiosemiBDFConverter
from Utilities.merger import Merger

conv = BiosemiBDFConverter(["./DataBDF/Snippets/Train",
                            "./DataBDF/Snippets/Val",
                            "./DataBDF/Snippets/Test"],
                           "./DataBDF/Out")
conv.convert_and_save()

#merger = Merger("./DataBDF/Out/", "./Data/EEGLarge/", "./DataMerged/")
#merger.merge()
