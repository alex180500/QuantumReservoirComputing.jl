using QuantumReservoirComputing
using MLDatasets: MNIST
using Random: seed!
seed!(260625)

train_x, train_y = MNIST(:train)[:]
test_x, test_y = MNIST(:test)[:]
train_data = reshape(train_x, 784, :)
test_data = reshape(test_x, 784, :)

N = 8
dim = 2^N
train_pca, test_pca = pca_analysis(train_data, test_data, 2 * N)

encoding = hcat(rescale_data(train_pca, test_pca)...)
q_states = dense_angle_encoding(encoding[1:N, :], encoding[(N + 1):end, :])

base_U = haar_unitary(dim)
reservoir_outputs = Float32.(qelm_compute(base_U, q_states))

results = nn_layer(reservoir_outputs, train_y, test_y; rate=0.1, epochs=100);
