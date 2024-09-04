import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, file_path):
        self.parquet_file = pq.ParquetFile(file_path)
        self.num_rows = self.parquet_file.metadata.num_rows
        self.data = pq.read_table(file_path).to_pandas()
        
    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        # Ensure idx is within bounds
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.data)}")
        
        # Access the row directly from the DataFrame
        row = self.data.iloc[idx]

        # Convert the row to tensors
        embedding1 = torch.tensor(row["embedding1"], dtype=torch.float32)
        embedding2 = torch.tensor(row["embedding2"], dtype=torch.float32)
        target = torch.tensor(row["target"], dtype=torch.float32)
        
        return embedding1, embedding2, target

# from torch.utils.data import DataLoader

# def process_batch(batch, model):
#     # Move batch to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     batch = batch.to(device)
    
#     # Process the batch with your model
#     model.eval()
#     with torch.no_grad():
#         output = model(batch)
    
#     # Do something with the output (e.g., save results, aggregate, etc.)
#     process_output(output)

# def main():
#     # Initialize your dataset
#     dataset = ParquetDataset('your_large_file.parquet')
    
#     # Create DataLoader
#     batch_size = 64  # Adjust based on your memory constraints and model requirements
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
#     # Load your PyTorch model
#     model = YourPyTorchModel()
#     model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
#     # Process batches
#     for batch in dataloader:
#         process_batch(batch, model)

# if __name__ == "__main__":
#     main()
