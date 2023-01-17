from Utilities.preconverters import LargeEEGDataPreConverter, BiosemiBDFPreConverter

preconv = BiosemiBDFPreConverter("./DataBDF/Kuba", "./DataBDF/Snippets")
preconv.preconvert()

#preconv = LargeEEGDataPreConverter("./DataEEG/", "./DataEEG/Snippets")
#preconv.preconvert()