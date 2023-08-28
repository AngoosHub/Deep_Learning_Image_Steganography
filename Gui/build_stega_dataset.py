from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from image import *



def create_stega_database():
    
    model = get_model()

    data_path = Path("data/stega_dataset/original")

    test_transform = transforms.Compose([
        transforms.Resize(size=(224)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    test_dataset = datasets.ImageFolder(root=data_path,
                                       transform=test_transform)
    
    test_dataloader = DataLoader(dataset=test_dataset, 
                                 batch_size=2, 
                                 num_workers=1,
                                 shuffle=False,
                                 drop_last=True)
    

    denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])
    pil_transfrom = transforms.ToPILImage()
    
    model.eval()

    image_num = 0
    
    for index, (data, label) in enumerate(test_dataloader):
        a, b = data.split(2//2,dim=0)
        cuda_cover = a.to(device)
        cuda_secret = b.to(device)

        with torch.inference_mode():
            modified_cover, recovered_secret = model(cuda_cover, cuda_secret)
        
        # cover = cuda_cover.cpu().squeeze(0)
        cover_x = modified_cover.cpu().squeeze(0)
        # secret = cuda_secret.cpu().squeeze(0)
        # secret_x = recovered_secret.cpu().squeeze(0)

        # plot_images_comparison(cover, cover_x, secret, secret_x, show_image=True)
        # test_plot(cover, cover_x, secret, secret_x, show_image=True)

        # cover_denorm = denormalize_and_toPIL(cover)
        # cover_x_denorm = denormalize_and_toPIL(cover_x)
        tensor_denorm = denorm(cover_x).clamp(min=0, max=1)
        cover_x_denorm = pil_transfrom(tensor_denorm)
        # secret_denorm = denormalize_and_toPIL(secret)
        # secret_x_denorm = denormalize_and_toPIL(secret_x)

        image_num_str = str(image_num).zfill(8)
        cover_x_denorm.save(f"data/stega_dataset/train/Stega_{image_num_str}.JPEG")
        image_num += 1



    print("Done")



if __name__ == "__main__":
    create_stega_database()


