import torch
import torch.multiprocessing as mp

# Assuming preprocess_function is defined here or imported
def preprocess_function(data):
    # Dummy preprocessing function
    return data * 2  # Replace this with your actual preprocessing logic

def preprocess_data(worker_id, input_queue, output_queue):
    # Ensure the worker_id does not exceed the available GPUs
    torch.cuda.set_device(worker_id)
    
    while True:
        data = input_queue.get()
        if data is None:
            break
        
        # Ensure preprocess_function is defined or accessible
        processed_data = preprocess_function(data).to(f'cuda:{worker_id}')
        output_queue.put(processed_data)

def main():
    mp.set_start_method('spawn')
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    num_gpus = torch.cuda.device_count()
    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=preprocess_data, args=(i, input_queue, output_queue))
        p.start()
        processes.append(p)

    # Your data processing logic continues here...

if __name__ == '__main__':
    main()
