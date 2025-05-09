# fast average values of pauli matrices of a qubit (or 2)
avg_z(ρ::Matrix{<:Number}) = real(ρ[1, 1] - ρ[2, 2])
avg_x(ρ::Matrix{<:Number}) = real(ρ[1, 2] + ρ[2, 1])
avg_y(ρ::Matrix{<:Number}) = real(im * (ρ[1, 2] - ρ[2, 1]))
avg_zz(ρ::Matrix{<:Number}) = real(ρ[1, 1] - ρ[2, 2] - ρ[3, 3] + ρ[4, 4])

# calculates the average z component of a quantum state
# given a density matrix ρ and multiple simulated measurement
function avg_z_finite(
    ρ::AbstractMatrix{T},
    trials::Integer=1_000_000,
    rng::AbstractRNG=default_rng()
) where {T<:Number}
    prob_up = real(ρ[1, 1])
    net_spin = count(rand(rng) < prob_up for _ in 1:trials)
    return 2 * net_spin / trials - 1
end
