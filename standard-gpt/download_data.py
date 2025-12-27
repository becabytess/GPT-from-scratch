import os
from datasets import load_dataset
from tqdm import tqdm

#download 10 billion tokens of internet text


def download_openwebtext():
    
    
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)

    output_file = 'openwebtext.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        count = 0
        total_chars = 0
        
        for sample in tqdm(ds):
            text = sample['text']
           
            f.write(text + '\n\n')
            
            total_chars += len(text)
            count += 1
           
                
    print(f"\nDone!")
    print(f"Saved to: {output_file}")
    print(f"Total documents: {count}")
    print(f"Total characters: {total_chars}")
    print(f"Approx size: {total_chars / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    download_openwebtext()

