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

function time_update(x, P::Cholesky, F, Q::Cholesky)
    x_apriori = F * x
    Q, R = qr(Hcat(F * P.L, Q.L)')
    P_apriori = Cholesky(R, 'U', 0)
    SRKFTimeUpdate(x_apriori, P_apriori)
end

function measurement_update(x, P::Cholesky, y, H, R::Cholesky)
    ỹ = y .- H * x
    dim_y = length(y)
    dim_x = length(x)
    M = [R.U        zeros(dim_y, dim_x)
         H * P.U    P.U                 ]
    Q, R = qr(M)
    PHᵀ = Cholesky(R[1:dim_y, dim_y + 1:end], 'U', 0)
    S = Cholesky(R[1:dim_y, 1:dim_y], 'U', 0)
    K = PHᵀ.L / S.L
    x_post = x .+ K * ỹ
    P_post = Cholesky(R[dim_y + 1:end, dim_y + 1:end], 'U', 0)
    SRKFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end
