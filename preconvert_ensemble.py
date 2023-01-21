from Utilities.preconverters_ensemble import LargeEEGDataPreConverter, BiosemiBDFPreConverter

preconv = BiosemiBDFPreConverter("./DataBDF/", "./DataBDF/Snippets")
preconv.preconvert()

#preconv = LargeEEGDataPreConverter("./DataEEG/", "./DataEEG/Snippets")
#preconv.preconvert()