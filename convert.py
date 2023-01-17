from Utilities.converters import LargeEEGDataConverter, BiosemiBDFConverter
from Utilities.merger import Merger

conv = BiosemiBDFConverter(["./DataBDF/Snippets/KubaTrain/",
                            "./DataBDF/Snippets/KubaVal/",
                            "./DataBDF/Snippets/KubaTest/"],
                           "./DataBDF/OutKuba/")
conv.convert_and_save()

#merger = Merger("./DataBDF/Out/", "./Data/EEGLarge/", "./DataMerged/")
#merger.merge()
