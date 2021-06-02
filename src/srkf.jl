struct SRKFTUIntermediate{T}
    x_apri::Vector{T}
    p_apri::Matrix{T}
    qr_zeros::Vector{T}
    qr_space::Vector{T}
    puft_vcat_q::Matrix{T}
end

function SRKFTUIntermediate(T::Type, num_x::Number)
    qr_zeros = zeros(T, 2 * num_x)
    puft_vcat_q = Matrix{T}(undef, 2 * num_x, num_x)
    qr_space_length = calc_gels_working_size(puft_vcat_q, qr_zeros)
    SRKFTUIntermediate(
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x),
        qr_zeros,
        Vector{T}(undef, qr_space_length),
        puft_vcat_q
    )
end

SRKFTUIntermediate(num_x::Number) = SRKFTUIntermediate(Float64, num_x)

function calc_upper_triangular_of_qr!(A)
    R, = LAPACK.gels!('N', A, zeros(size(A, 1), 1)) # Q, R = qr(A)
    R
end

function calc_upper_triangular_of_qr_inplace!(res, A, B, space)
    mygels!(res, A, B, space)
end

# This implementation is based on
# https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/square_root.py
@inline function calc_apriori_covariance(
    P::Cholesky,
    F::Union{Number, AbstractMatrix},
    Q::Cholesky
)
    A = vcat(P.U * F', Q.U)
    R = calc_upper_triangular_of_qr!(A)
    Cholesky(R, 'U', 0)
end

function time_update!(tu::SRKFTUIntermediate, x, P::Cholesky, F::Union{Number, AbstractMatrix}, Q::Cholesky)
    x_apri = calc_apriori_state!(tu.x_apri, x, F)
    tu.puft_vcat_q[1:size(F, 1),:] .= @~ P.U * F'
    tu.puft_vcat_q[size(F, 1) + 1:end,:] .= Q.U
    R = calc_upper_triangular_of_qr_inplace!(tu.p_apri, tu.puft_vcat_q, tu.qr_zeros, tu.qr_space)
    P_apri = Cholesky(R, 'U', 0)
    KFTimeUpdate(x_apri, P_apri)
end

function measurement_update(
    x,
    P::Cholesky,
    y,
    H::Union{Number, AbstractVector, AbstractMatrix},
    R::Cholesky
)
    typeof(x) <: ConsideredState && @warn "The lower right part of P (P_cc - the considered part) is not calculated correctly. Please check the algorithm."
    ỹ = calc_innovation(H, x, y)
    PHᵀ, S, P_post = calc_cross_cov_innovation_posterior(P, H, R)
    K = calc_kalman_gain(PHᵀ, S.L)
    x_post = calc_posterior_state(x, K, ỹ)
    KFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

function create_matrix_for_qr(P::Cholesky, H, R::Cholesky)
    dim_y = size(R, 1)
    dim_x = size(P, 1)
    M = zeros(dim_y + dim_x, dim_y + dim_x)
    M[1:dim_y, 1:dim_y] .= R.U
    M[dim_y + 1:end, 1:dim_y] .= P.U * H'
    M[dim_y + 1:end, dim_y + 1:end] .= P.U
    M
end

function calc_cross_cov_innovation_posterior(P::Cholesky, H, R::Cholesky)
    dim_y = size(R, 1)
    M = create_matrix_for_qr(P, H, R)
    RU = calc_upper_triangular_of_qr!(M)
    PHᵀ = transpose(RU[1:dim_y, dim_y + 1:end])
    S = Cholesky(RU[1:dim_y, 1:dim_y], 'U', 0)
    P_post = Cholesky(RU[dim_y + 1:end, dim_y + 1:end], 'U', 0)
    PHᵀ, S, P_post
end

struct SRKFMUIntermediate{T,K<:Union{<:AbstractVector{T},<:AbstractMatrix{T}}}
    innovation::Vector{T}
    kalman_gain::K
    m::Matrix{T}
    qr_zeros::Vector{T}
    qr_space::Vector{T}
    R::Matrix{T}
    x_posterior::Vector{T}
end

function SRKFMUIntermediate(T::Type, num_x::Number, num_y::Number)
    qr_zeros = zeros(T, num_x + num_y)
    M = Matrix{T}(undef, num_x + num_y, num_x + num_y)
    qr_space_length = calc_gels_working_size(M, qr_zeros)
    SRKFMUIntermediate(
        Vector{T}(undef, num_y),
        Matrix{T}(undef, num_x, num_y),
        M,
        qr_zeros,
        Vector{T}(undef, qr_space_length),
        Matrix{T}(undef, num_x + num_y, num_x + num_y),
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
    M = mu.m
    M[1:dim_y, 1:dim_y] .= R.U
    M[dim_y + 1:end, 1:dim_y] .= @~ P.U * H'
    M[dim_y + 1:end, dim_y + 1:end] .= P.U
    RU = calc_upper_triangular_of_qr_inplace!(mu.R, M, mu.qr_zeros, mu.qr_space)
    PHᵀ = transpose(@view(RU[1:dim_y, dim_y + 1:end]))
    S = Cholesky(@view(RU[1:dim_y, 1:dim_y]), 'U', 0)
    K = calc_kalman_gain!(mu.kalman_gain, PHᵀ, S.L)
    x_post = calc_posterior_state!(mu.x_posterior, x, K, ỹ)
    P_post = Cholesky(@view(RU[dim_y + 1:end, dim_y + 1:end]), 'U', 0)
    KFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

function calc_kalman_gain!(K, PHᵀ, SL::LowerTriangular)
    K .= PHᵀ
    rdiv!(K, SL)
end