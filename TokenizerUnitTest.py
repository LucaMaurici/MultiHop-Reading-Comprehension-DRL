from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor

loaded_data = ["now this ain't funny", "so don't you dare laugh now", "bum bum ehi ehi, , !"]
encoder = StaticTokenizerEncoder(loaded_data, tokenize=lambda s: s.split())
#encoded_data = [encoder.encode(example) for example in loaded_data]
example_data = ['ehi, ehi ciao', 'funny']
encoded_data = [encoder.encode(example) for example in example_data]

print(encoded_data)