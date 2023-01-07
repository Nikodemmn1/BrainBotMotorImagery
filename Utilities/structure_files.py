import os
import fnmatch

SOURCE_PATH = "../DataBDF/TestData/"
file_name_stems = {'kub': 'Kuba',
                   'nik': 'Nikodem',
                   'pio': 'Piotr'}
DESTINATION_PATH = {'kub': "../DataBDF/TestData/Kuba/",
                    'nik': "../DataBDF/TestData/Nikodem/",
                    'pio': "../DataBDF/TestData/Piotr/"}


def main():
    files = os.listdir(SOURCE_PATH)
    files_w_subject = []
    for file in files:
        if os.path.isdir(SOURCE_PATH + file):
            continue
        if fnmatch.fnmatch(file, "*[Kk]ub*"):
            files_w_subject.append(('kub', file))
        elif fnmatch.fnmatch(file, "*[Nn]ik*"):
            files_w_subject.append(('nik', file))
        elif fnmatch.fnmatch(file, "*[Pp]io*"):
            files_w_subject.append(('pio', file))

    counter = {'kub': 0,
               'nik': 0,
               'pio': 0}

    for subject, file in files_w_subject:
        os.rename(SOURCE_PATH + file, DESTINATION_PATH[subject] + file_name_stems[subject] + '_' + str(counter[subject]) + '.bdf')
        counter[subject] += 1


if __name__ == "__main__":
    main()
