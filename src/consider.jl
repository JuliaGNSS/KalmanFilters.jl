@inline function calc_kalman_gain(PHᵀ, S, consider::AbstractVector)
    not_consider = filter(i -> !(i in consider), 1:size(PHᵀ, 1))
    PHᵀ[not_consider, :] / S
end

@inline function calc_posterior_state(x, x̃, consider::AbstractVector)
    not_consider = filter(i -> !(i in consider), 1:length(x))
    x_post = copy(x)
    x_post[not_consider] = x[not_consider] + x̃
    x_post
end

@inline function calc_posterior_state(x, K, ỹ, consider::AbstractVector)
    calc_posterior_state(x, K * ỹ, consider::AbstractVector)
end

@inline function calc_posterior_covariance(P, PHᵀ, K, consider::AbstractVector)
    not_consider = filter(i -> !(i in consider), 1:size(PHᵀ, 1))
    P_post = copy(P)
    P_post[not_consider, not_consider] =
        P[not_consider, not_consider] .- PHᵀ[not_consider, :] * K'
    P_post[consider, not_consider] = P[consider, not_consider] .- PHᵀ[consider, :] * K'
    P_post[not_consider, consider] = P_post[consider, not_consider]'
    P_post
end

@inline function extract_posterior_covariance(RU, num_y, P, consider::AbstractVector)
    not_consider = filter(i -> !(i in consider), 1:size(P, 1))
    P_post = copy(P)
    num_not_consider = size(P, 1) - length(consider)
    P_post.factors[not_consider, :] = RU[(num_y+1):(num_y+num_not_consider), (num_y+1):end]
    reconstruct_consider_covariance!(P_post, P, consider)
    P_post
end

function calc_kalman_gain_and_posterior_covariance(
    P::Cholesky,
    Pᵪᵧ,
    S::Cholesky,
    consider::AbstractVector,
)
    not_consider = filter(i -> !(i in consider), 1:size(P, 1))
    U = S.uplo === 'U' ? Pᵪᵧ / S.U : Pᵪᵧ / S.L'
    K = S.uplo === 'U' ? U[not_consider, :] / S.U' : U[not_consider, :] / S.L
    P_post = copy(P)
    P_xx_xc = reduce(lowrankdowndate, eachcol(U); init = P)
    P_post.factors[not_consider, :] = P_xx_xc.U[not_consider, :]
    reconstruct_consider_covariance!(P_post, P, consider)
    K, P_post
end

function reconstruct_consider_covariance!(P_post, P, consider)
    not_consider = filter(i -> !(i in consider), 1:size(P, 1))
    B = P_post.U[not_consider, not_consider] * P_post.U[not_consider, consider]
    P_cc = Cholesky(P.factors[consider, consider], P.uplo, 0)
    P_cc_updated = reduce(
        lowrankupdate,
        eachcol(collect(P.factors[not_consider, consider]'));
        init = P_cc,
    )
    P_cc_downdated = reduce(
        lowrankdowndate,
        eachcol(B' / P_post.L[not_consider, not_consider]);
        init = P_cc_updated,
    )
    P_post.factors[consider, consider] = P_cc_downdated.U
    P_post.factors[consider, not_consider] = P_post.factors[not_consider, consider]'
end
