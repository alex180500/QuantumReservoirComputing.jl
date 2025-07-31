using QuantumReservoirComputing

test = LocalOperators{5,ComplexF64}(undef)

testx = LocalOperators(pauli_x, 5)
testx2 = LocalOperators{5}(pauli_x)

xx_monroe_pbc(testx; α=1.0)
xx_monroe_obc(testx2; α=1.0)
