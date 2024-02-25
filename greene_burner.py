import torch

if __name__ == "__main__":
    duration_seconds = 60
    try:
        torch.cuda.set_device(0)
        tensor = torch.rand((1000, 1000), device='cuda')
        target_iterations = int(1e8)
        print(f"Starting on GPU {0}...")
        for i in range(target_iterations):
            tensor = torch.matmul(tensor, tensor)
        print(f"Pausing after {duration_seconds} seconds...")
        import time
        time.sleep(duration_seconds)
        print("Done.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Cleaning up...")
        exit()