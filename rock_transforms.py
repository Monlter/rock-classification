from torchvision.transforms import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
    # transforms.RandomGrayscale(0.2),
    # transforms.RandomCrop(1024),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5),
    #                      (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    # transforms.CenterCrop((1024, 1024)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5),
    #                      (0.5, 0.5, 0.5)),
])
