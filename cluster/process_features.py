with open('sample_features_is09.txt', 'r') as f:
    sample_features = f.read()
    num_features_is09 = 0
    for line in sample_features.splitlines():
        if line.startswith('@attribute'):
            num_features_is09 += 1

print(f"Number of is09 features: {num_features_is09}")

with open('sample_features_egemaps.txt', 'r') as f:
    sample_features = f.read()
    num_features_egemaps = 0
    for line in sample_features.splitlines():
        if line.startswith('@attribute'):
            num_features_egemaps += 1

print(f"Number of egemaps features: {num_features_egemaps}")
print(f"Total number of features: {num_features_is09 + num_features_egemaps}")

