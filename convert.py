from Utilities.preconverters import LargeEEGDataPreConverter, BiosemiBDFPreConverter
from Utilities.converters import LargeEEGDataConverter, BiosemiBDFConverter

#preconv = BiosemiBDFPreConverter("./DataBDF/", "./DataBDF/Snippets")
#preconv.preconvert()

conv = BiosemiBDFConverter(["./DataBDF/Snippets/Train",
                            "./DataBDF/Snippets/Val",
                            "./DataBDF/Snippets/Test"],
                           "./DataBDF/Out")
conv.convert_and_save()

#preconv = LargeEEGDataPreConverter("./DataEEG/", "./DataEEG/Snippets")
#preconv.preconvert()
#
#conv = LargeEEGDataConverter(["./DataEEG/Snippets/Train",
#                            "./DataEEG/Snippets/Val",
#                            "./DataEEG/Snippets/Test"],
#                           "./DataEEG/Out")
#conv.convert_and_save()
