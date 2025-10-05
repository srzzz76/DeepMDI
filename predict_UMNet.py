import glob
import numpy as np
import torch
import os
import cv2
from NAFNet_arch import NAFNet_arch
import re

if __name__ == "__main__":
    # -------------------- Configuration --------------------
    model_name = 'UMNet'           # Model name
    test_folder = 'PDNet'          # Test data folder
    data_folder = 'simu_test'  # Data folder

    base_path = f'autodl-fs/{data_folder}'
    tests_path1 = glob.glob(f'{base_path}/{test_folder}/frame1_n/*.png')
    tests_path1_sorted = sorted(tests_path1, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))

    print("=== Path Configuration ===")
    print(f"Model name: {model_name}")
    print(f"Data folder: {data_folder}")
    print(f"Test folder: {test_folder}")
    print(f"Number of test images found: {len(tests_path1_sorted)}\n")

    # -------------------- Initialize Network --------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = NAFNet_arch(in_channel=2, out_channel=2, width=16, middle_blk_num=1,
                      enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1])
    net.to(device)
    net.load_state_dict(torch.load(f'model/{model_name}.pth', map_location=device, weights_only=True))
    net.eval()

    # -------------------- Prediction --------------------
    for test_path in tests_path1_sorted:
        test_path2 = test_path.replace('frame1_n', 'frame2_n')
        save_res_path1 = test_path.replace(f'{test_folder}/frame1_n', f'test_{model_name}/wrapped')

        # Create output directory if not exist
        os.makedirs(os.path.dirname(save_res_path1), exist_ok=True)

        # Load and normalize images
        img1 = cv2.imread(test_path, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(test_path2, cv2.IMREAD_UNCHANGED)

        img1 = img1 / 65535.0 if img1.max() > 255 else img1 / 255.0
        img2 = img2 / 65535.0 if img2.max() > 255 else img2 / 255.0

        # Reshape and concatenate to 4D tensor
        img1 = img1.reshape(1, 1, img1.shape[0], img1.shape[1])
        img2 = img2.reshape(1, 1, img2.shape[0], img2.shape[1])
        img_tensor = np.concatenate((img1, img2), axis=1)
        img_tensor = torch.from_numpy(img_tensor).to(device=device, dtype=torch.float32)

        # Forward pass
        with torch.no_grad():
            pred = net(img_tensor)

        # Process prediction: only first channel (frame1)
        pred_img = pred[:, 0, :, :].unsqueeze(1).detach().cpu().numpy()[0, 0] + 0.5

        # Save prediction as 16-bit PNG
        pred_16bit = np.clip(pred_img * 65535, 0, 65535).astype(np.uint16)
        cv2.imwrite(save_res_path1, pred_16bit)

        print(f"Saved: {save_res_path1}")

    print("All predictions saved successfully.")
