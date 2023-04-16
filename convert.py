from Utilities.converters import LargeEEGDataConverter, BiosemiBDFConverter

conv = BiosemiBDFConverter(["./DataBDF/Snippets/Train",
                            "./DataBDF/Snippets/Val",
                            "./DataBDF/Snippets/Test"],
                           "./DataBDF/OutNikodem")
conv.convert_and_save()

#conv = LargeEEGDataConverter(["./DataEEG/Snippets/Train",
#                            "./DataEEG/Snippets/Val",
#                            "./DataEEG/Snippets/Test"],
#                           "./DataEEG/OutNikodem")
#conv.coanvert_and_save()
