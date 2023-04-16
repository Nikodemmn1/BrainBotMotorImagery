from functions import get_dataloaders

def main():
    included_classes = [0, 1, 2]
    included_channels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    source_dataset_prefix = "../DataBDF/OutKuba_full_dataset/OutKuba_full_dataset"
    target_dataset_prefix = "../DataBDF/OutKuba_full_dataset/OutKuba_full_dataset"

    source_dataloaders = get_dataloaders(source_dataset_prefix, included_classes, included_channels)
    target_dataloaders = get_dataloaders(target_dataset_prefix, included_classes, included_channels)

    print(source_dataloaders)

main()