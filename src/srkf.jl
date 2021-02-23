struct SRKFTimeUpdate{X,P} <: AbstractSRTimeUpdate
    state::X
    covariance::P
end

struct SRKFMeasurementUpdate{X,P,R,S,K} <: AbstractSRMeasurementUpdate
    state::X
    covariance::P
    innovation::R
    innovation_covariance::S
    kalman_gain::K
end

struct SRKFTUIntermediate{T}
    x_apri::Vector{T}
    p_apri::Matrix{T}
    zeros::Vector{T}
    puft_vcat_q::Matrix{T}
end

SRKFTUIntermediate(T::Type, num_x::Number) =
    SRKFTUIntermediate(
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x),
        zeros(T, 2 * num_x),
        Matrix{T}(undef, 2 * num_x, num_x)
    )

SRKFTUIntermediate(num_x::Number) = SRKFTUIntermediate(Float64, num_x)

function calc_upper_triangular_of_qr!(A)
    R, = LAPACK.gels!('N', A, zeros(size(A, 1), 1)) # Q, R = qr(A)
    R
end

function calc_upper_triangular_of_qr_inplace!(A, B)
    R, = LAPACK.gels!('N', A, B)
    R
end

function time_update(x, P::Cholesky, F::Union{Number, AbstractMatrix}, Q::Cholesky)
    x_apri = calc_apriori_state(x, F)
    A = vcat(P.U * F', Q.U)
    R = calc_upper_triangular_of_qr!(A)
    P_apri = Cholesky(R, 'U', 0)
    SRKFTimeUpdate(x_apri, P_apri)
end

function time_update!(tu::SRKFTUIntermediate, x, P::Cholesky, F::Union{Number, AbstractMatrix}, Q::Cholesky)
    x_apri = calc_apriori_state!(tu.x_apri, x, F)
    tu.puft_vcat_q[1:size(F, 1),:] .= @~ P.U * F'
    tu.puft_vcat_q[size(F, 1) + 1:end,:] .= Q.U
    R = calc_upper_triangular_of_qr_inplace!(tu.puft_vcat_q, tu.zeros)
    P_apri = Cholesky(R, 'U', 0)
    SRKFTimeUpdate(x_apri, P_apri)
end

function measurement_update(
    x,
    P::Cholesky,
    y,
    H::Union{Number, AbstractVector, AbstractMatrix},
    R::Cholesky
)
    ỹ = calc_innovation(H, x, y)
    dim_y = length(y)
    dim_x = length(x)
    M = zeros(dim_y + dim_x, dim_y + dim_x)
    M[1:dim_y, 1:dim_y] .= R.U
    M[dim_y + 1:end, 1:dim_y] .= (H * P.L)'
    M[dim_y + 1:end, dim_y + 1:end] .= P.U
    R = calc_upper_triangular_of_qr!(M)
    PHᵀ = transpose(@view(R[1:dim_y, dim_y + 1:end]))
    S = Cholesky(@view(R[1:dim_y, 1:dim_y]), 'U', 0)
    K = calc_kalman_gain(PHᵀ, S.L)
    x_post = calc_posterior_state(x, K, ỹ)
    P_post = Cholesky(@view(R[dim_y + 1:end, dim_y + 1:end]), 'U', 0)
    SRKFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

struct SRKFMUIntermediate{T,K<:Union{<:AbstractVector{T},<:AbstractMatrix{T}}}
    innovation::Vector{T}
    kalman_gain::K
    m::Matrix{T}
    zeros::Vector{T}
    x_posterior::Vector{T}
end

function SRKFMUIntermediate(T::Type, num_x::Number, num_y::Number)
    return SRKFMUIntermediate(
        Vector{T}(undef, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_x + num_y, num_x + num_y),
        Vector{T}(undef, num_x + num_y),
        Vector{T}(undef, num_x)
    )
end

SRKFMUIntermediate(num_x::Number, num_y::Number) = SRKFMUIntermediate(Float64, num_x, num_y)

function measurement_update!(
    mu::SRKFMUIntermediate,
    x,
    P::Cholesky,
    y,
    H::Union{Number, AbstractVector, AbstractMatrix},
    R::Cholesky
)
    ỹ = calc_innovation!(mu.innovation, H, x, y)
    dim_y = length(y)
    dim_x = length(x)
    M = mu.m
    M[1:dim_y, 1:dim_y] .= R.U
    M[dim_y + 1:end, 1:dim_y] .= @~ P.U * H'
    M[dim_y + 1:end, dim_y + 1:end] .= P.U
    R = calc_upper_triangular_of_qr_inplace!(M, mu.zeros)
    PHᵀ = transpose(@view(R[1:dim_y, dim_y + 1:end]))
    S = Cholesky(@view(R[1:dim_y, 1:dim_y]), 'U', 0)
    K = calc_kalman_gain!(mu.kalman_gain, PHᵀ, S.L)
    x_post = calc_posterior_state!(mu.x_posterior, x, K, ỹ)
    P_post = Cholesky(@view(R[dim_y + 1:end, dim_y + 1:end]), 'U', 0)
    SRKFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

function calc_kalman_gain!(K, PHᵀ, SL::LowerTriangular)
    K .= PHᵀ
    rdiv!(K, SL)
end