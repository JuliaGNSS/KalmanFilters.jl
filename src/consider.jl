struct ConsideredState{T, V <: AbstractVector{T}} <: AbstractVector{T}
    x::V
    x_c::V
end

@inline Base.size(C::ConsideredState) = (length(C.x) + length(C.x_c), )
@inline function Base.getindex(C::ConsideredState, inds::Vararg{Int,1})
    @boundscheck checkbounds(C, inds[1])
    @inbounds if inds[1] <= length(C.x)
        C.x[inds[1]]
    else
        C.x_c[inds[1] - length(C.x)]
    end
end

struct ConsideredCovariance{T, M <: AbstractMatrix{T}, CM <: AbstractMatrix{T}} <: AbstractMatrix{T}
    P::M
    P_xc::CM
    P_cc::M
    ConsideredCovariance{T, M, CM}(P, P_xc, P_cc) where {T <: Real, M <: AbstractMatrix{T}, CM <: AbstractMatrix{T}} =
        size(P, 1) == size(P, 2) == size(P_xc, 1) && size(P_cc, 1) == size(P_cc, 2) == size(P_xc, 2) ?
        new{T, M, CM}(P, P_xc, P_cc) :
        error("Matrices do not have correct sizes [P P_xc; P_xc' P_cc]")
end

function ConsideredCovariance(
    P::M,
    P_xc::CM,
    P_cc::M
) where {T, M <: AbstractMatrix{T}, CM <: AbstractMatrix{T}}
    ConsideredCovariance{T, M, CM}(P, P_xc, P_cc)
end

function LinearAlgebra.cholesky(C::ConsideredCovariance)
    P_chol = cholesky(C.P)
    Q = C.P_cc - C.P_xc' * (C.P \ C.P_xc)
    Q_chol = cholesky(Hermitian(Q))
    Cholesky(ConsideredCovariance(P_chol.U, P_chol.L \ C.P_xc, Q_chol.U), 'U', 0)
end

@inline Base.size(C::ConsideredCovariance) = size(C.P) .+ size(C.P_cc)
@inline function Base.getindex(C::ConsideredCovariance, inds::Vararg{Int,2})
    @boundscheck checkbounds(C, inds[1], inds[2])
    @inbounds if inds[1] <= size(C.P, 1) && inds[2] <= size(C.P, 2)
        C.P[inds[1], inds[2]]
    elseif inds[1] > size(C.P, 1) && inds[2] <= size(C.P, 2)
        adjoint(C.P_xc)[inds[1] - size(C.P, 1), inds[2]]
    elseif inds[1] <= size(C.P, 1) && inds[2] > size(C.P, 2)
        C.P_xc[inds[1], inds[2] - size(C.P, 2)]
    else
        C.P_cc[inds[1] - size(C.P, 1), inds[2] - size(C.P, 2)]
    end
end

struct ConsideredCrossCovariance{T, M <: AbstractMatrix{T}}
    P_xy::M
    P_xyc::M
end

struct ConsideredProcessModel{T, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
    F::M
    F_xc::M
    ConsideredProcessModel{T, M}(F, F_xc) where {T <: Real, M <: AbstractMatrix{T}} =
        size(F, 1) == size(F, 2) == size(F_xc, 1) ?
        new{T, M}(F, F_xc) :
        error("Matrices do not have correct sizes [F F_xc; 0 I]")
end

function ConsideredProcessModel(
    F::M,
    F_xc::M,
) where {T <: Real, M <: AbstractMatrix{T}}
    ConsideredProcessModel{T, M}(F, F_xc)
end

@inline Base.size(C::ConsideredProcessModel) =
    (size(C.F, 2) + size(C.F_xc, 2), size(C.F, 2) + size(C.F_xc, 2))
@inline function Base.getindex(C::ConsideredProcessModel{T}, inds::Vararg{Int,2}) where T
    @boundscheck checkbounds(C, inds[1], inds[2])
    @inbounds if inds[1] <= size(C.F, 1) && inds[2] <= size(C.F, 2)
        C.F[inds[1], inds[2]]
    elseif inds[1] <= size(C.F, 1) && inds[2] > size(C.F, 2)
        C.F_xc[inds[1], inds[2] - size(C.F, 2)]
    elseif inds[1] == inds[2]
        one(T)
    else
        zero(T)
    end
end

struct ConsideredMeasurementModel{T, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
    H::M
    H_c::M
    ConsideredMeasurementModel{T, M}(H, H_c) where {T <: Real, M <: AbstractMatrix{T}} =
        size(H, 1) == size(H_c, 1) ?
        new{T, M}(H, H_c) :
        error("Matrices do not have correct sizes [H H_c]")
end

function ConsideredMeasurementModel(
    H::M,
    H_c::M,
) where {T <: Real, M <: AbstractMatrix{T}}
    ConsideredMeasurementModel{T, M}(H, H_c)
end

@inline Base.size(C::ConsideredMeasurementModel) =
    (size(C.H, 1), size(C.H, 2) + size(C.H_c, 2))
@inline function Base.getindex(C::ConsideredMeasurementModel, inds::Vararg{Int,2})
    @boundscheck checkbounds(C, inds[1], inds[2])
    @inbounds if inds[2] <= size(C.H, 2)
        C.H[inds[1], inds[2]]
    else
        C.H_c[inds[1], inds[2] - size(C.H, 2)]
    end
end

@inline function calc_apriori_state(
    cons_state::ConsideredState,
    cons_process::ConsideredProcessModel
)
    x_apri = cons_process.F * cons_state.x + cons_process.F_xc * cons_state.x_c
    ConsideredState(x_apri, cons_state.x_c)
end

@inline function calc_apriori_covariance(
    cons_cov::ConsideredCovariance,
    cons_process::ConsideredProcessModel,
    Q
)
    P_apri = cons_process.F * cons_cov.P * cons_process.F' .+ Q .+
        cons_process.F * cons_cov.P_xc * cons_process.F_xc' .+
        cons_process.F_xc * cons_cov.P_xc' * cons_process.F' .+
        cons_process.F_xc * cons_cov.P_cc * cons_process.F_xc'
    P_xc_apri = cons_process.F * cons_cov.P_xc .+ cons_process.F_xc * cons_cov.P_cc
    ConsideredCovariance(P_apri, P_xc_apri, cons_cov.P_cc)
end

@inline function calc_apriori_covariance(
    cons_cov::Cholesky{<: Number, <: ConsideredCovariance},
    cons_proc::ConsideredProcessModel,
    Q::Cholesky
)
    P = cons_cov.factors.P
    P_xc = cons_cov.factors.P_xc
    P_cc = cons_cov.factors.P_cc
    F = cons_proc.F
    F_xc = cons_proc.F_xc
    A = [
        P * F' + P_xc * F_xc'   P_xc
        P_cc * F_xc'            P_cc
        Q.U                     zeros(size(Q, 1), size(P_cc, 2))
    ]
    R = calc_upper_triangular_of_qr!(A)
    Cholesky(ConsideredCovariance(
        UpperTriangular(R[1:size(P, 1), 1:size(P, 2)]),
        R[1:size(P, 1), size(P, 2) + 1:end],
        UpperTriangular(R[size(P, 1) + 1:end, size(P, 2) + 1:end]),
    ), 'U', 0)
end

@inline function calc_innovation(
    cons_meas::ConsideredMeasurementModel,
    cons_state::ConsideredState,
    y
)
    y .- cons_meas.H * cons_state.x .- cons_meas.H_c * cons_state.x_c
end

@inline function calc_P_xy(
    cons_cov::ConsideredCovariance,
    cons_meas::ConsideredMeasurementModel
)
    P_xy = cons_cov.P * cons_meas.H' .+ cons_cov.P_xc * cons_meas.H_c'
    P_xyc = cons_cov.P_xc' * cons_meas.H' .+ cons_cov.P_cc * cons_meas.H_c'
    ConsideredCrossCovariance(P_xy, P_xyc)
end

@inline function calc_kalman_gain(
    cons_cross_cov::ConsideredCrossCovariance,
    S
)
    cons_cross_cov.P_xy / S
end

@inline function calc_innovation_covariance(
    cons_meas::ConsideredMeasurementModel,
    cons_cov::ConsideredCovariance,
    R
)
    cons_meas.H * cons_cov.P * cons_meas.H' .+ R .+
    cons_meas.H * cons_cov.P_xc * cons_meas.H_c' .+
    cons_meas.H_c * cons_cov.P_xc' * cons_meas.H' .+
    cons_meas.H_c * cons_cov.P_cc * cons_meas.H_c'
end

@inline function calc_posterior_state(
    cons_state::ConsideredState,
    K,
    ỹ
)
    x_post = cons_state.x + K * ỹ
    ConsideredState(x_post, cons_state.x_c)
end

@inline function calc_posterior_covariance(
    cons_cov::ConsideredCovariance,
    cons_cross_cov::ConsideredCrossCovariance,
    K
)
    P_post = cons_cov.P .- cons_cross_cov.P_xy * K'
    P_xc_post = cons_cov.P_xc .- K * cons_cross_cov.P_xyc'
    ConsideredCovariance(P_post, P_xc_post, cons_cov.P_cc)
end

function create_matrix_for_qr(
    cons_cov::Cholesky{<: Number, <: ConsideredCovariance},
    cons_meas::ConsideredMeasurementModel,
    R::Cholesky
)
    P = cons_cov.factors.P
    P_xc = cons_cov.factors.P_xc
    P_cc = cons_cov.factors.P_cc
    H = cons_meas.H
    H_c = cons_meas.H_c
    [
        R.U                     zeros(size(R, 1), size(cons_cov, 2))
        P * H' + P_xc * H_c'    P                                   P_xc
        P_cc * H_c'             zeros(size(P_cc, 1), size(P, 2))    P_cc
    ]
end

function calc_cross_cov_innovation_posterior(
    cons_cov::Cholesky{<: Number, <: ConsideredCovariance},
    cons_meas::ConsideredMeasurementModel,
    R::Cholesky
)
    P = cons_cov.factors.P
    dim_y = size(R, 1)
    M = create_matrix_for_qr(cons_cov, cons_meas, R)
    RU = calc_upper_triangular_of_qr!(M)
    cons_cross_cov = ConsideredCrossCovariance(
        transpose(RU[1:dim_y, dim_y + 1:dim_y + size(P, 1)]),
        transpose(RU[1:dim_y, dim_y + size(P, 1) + 1:end])
    )
    S = Cholesky(RU[1:dim_y, 1:dim_y], 'U', 0)
    cons_P_post = Cholesky(ConsideredCovariance(
        UpperTriangular(RU[dim_y + 1:dim_y + size(P, 1), dim_y + 1:dim_y + size(P, 2)]),
        RU[dim_y + 1:dim_y + size(P, 1), dim_y + size(P, 2) + 1:end],
        UpperTriangular(RU[dim_y + size(P, 1) + 1:end, dim_y + size(P, 2) + 1:end]),
    ), 'U', 0)
    cons_cross_cov, S, cons_P_post
end

function calc_kalman_gain_and_posterior_covariance(cons_cov::ConsideredCovariance, Pᵪᵧ, S)
    P = cons_cov.P
    cons_cross_cov = ConsideredCrossCovariance(Pᵪᵧ[1:size(P, 1),:], Pᵪᵧ[size(P, 1) + 1:end,:])
    K = calc_kalman_gain(cons_cross_cov, S)
    P_posterior = calc_posterior_covariance(cons_cov, cons_cross_cov, K)
    K, P_posterior
end

function calc_kalman_gain_and_posterior_covariance(cons_cov::Cholesky{<: Real, <:ConsideredCovariance}, Pᵪᵧ, S::Cholesky)
    P_chol = Cholesky(cons_cov.factors.P, 'U', 0)
    P_cc_chol = Cholesky(cons_cov.factors.P_cc, 'U', 0)
    cons_cross_cov = ConsideredCrossCovariance(Pᵪᵧ[1:size(P_chol, 1),:], Pᵪᵧ[size(P_chol, 1) + 1:end,:])
    U = S.uplo === 'U' ? cons_cross_cov.P_xy / S.U : cons_cross_cov.P_xy / transpose(S.L)
    K = S.uplo === 'U' ? U / transpose(S.U) : U / S.L

    U_P = S.uplo === 'U' ? Pᵪᵧ / S.U : Pᵪᵧ / transpose(S.L)
    P_post = reduce(lowrankdowndate, eachcol(U_P), init = cons_cov)
    B1 = P_post.U[1:4,1:4] * P_post.U[1:4,5:6]
    P_cc_chol_post_updated1 = reduce(lowrankupdate, eachcol(collect(cons_cov.factors.P_xc')), init = P_cc_chol)
    P_cc_chol_post_updated2 = reduce(lowrankdowndate, eachcol(B1' / P_post.L[1:4,1:4]), init = P_cc_chol_post_updated1)
    K, Cholesky(ConsideredCovariance(P_post.U[1:4,1:4], P_post.U[1:4,5:6], collect(P_cc_chol_post_updated2.U)), 'U', 0)
end