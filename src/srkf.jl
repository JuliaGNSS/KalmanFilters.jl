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

function time_update(x, P::Cholesky, F::Union{Number, AbstractMatrix}, Q::Cholesky)
    x_apriori = F * x
    Q, R = qr(Hcat(F * P.L, Q.L)')
    P_apriori = Cholesky(R, 'U', 0)
    SRKFTimeUpdate(x_apriori, P_apriori)
end

function measurement_update(
    x,
    P::Cholesky,
    y,
    H::Union{Number, AbstractVector, AbstractMatrix},
    R::Cholesky
)
    ỹ = y .- H * x
    dim_y = length(y)
    dim_x = length(x)
    M = zeros(dim_y + dim_x, dim_y + dim_x)
    M[1:dim_y, 1:dim_y] .= R.U
    M[dim_y + 1:end, 1:dim_y] .= (H * P.L)'
    M[dim_y + 1:end, dim_y + 1:end] .= P.U
    Q, R = qr(M)
    PHᵀ = transpose(R[1:dim_y, dim_y + 1:end])
    S = Cholesky(R[1:dim_y, 1:dim_y], 'U', 0)
    K = PHᵀ / S.L
    x_post = x .+ K * ỹ
    P_post = Cholesky(R[dim_y + 1:end, dim_y + 1:end], 'U', 0)
    SRKFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end
