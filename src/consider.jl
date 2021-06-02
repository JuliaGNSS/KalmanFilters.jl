@inline function calc_kalman_gain(PHᵀ, S, consider::AbstractVector)
    PHᵀ[Not(consider), :] / S
end

@inline function calc_posterior_state(x, K, ỹ, consider::AbstractVector)
    x_post = copy(x)
    x_post[Not(consider)] = x[Not(consider)] + K * ỹ
    x_post
end

@inline function calc_posterior_covariance(P, PHᵀ, K, consider::AbstractVector)
    P_post = copy(P)
    P_post[Not(consider),Not(consider)] = P[Not(consider),Not(consider)] .- PHᵀ[Not(consider),:] * K'
    P_post[consider,Not(consider)] = P[consider,Not(consider)] .- PHᵀ[consider,:] * K'
    P_post[Not(consider),consider] = P_post[consider,Not(consider)]'
    P_post
end

@inline function extract_posterior_covariance(RU, num_y, P, consider::AbstractVector)
    P_post = copy(P)
    num_not_consider = size(P, 1) - length(consider)
    P_post.factors[Not(consider),:] = RU[num_y + 1:num_y + num_not_consider, num_y + 1:end]
    reconstruct_consider_covariance!(P_post, P, consider)
    P_post
end

function calc_kalman_gain_and_posterior_covariance(P::Cholesky, Pᵪᵧ, S::Cholesky, consider::AbstractVector)
    U = S.uplo === 'U' ? Pᵪᵧ / S.U : Pᵪᵧ / transpose(S.L)
    K = S.uplo === 'U' ? U[Not(consider),:] / transpose(S.U) : U[Not(consider),:] / S.L
    P_post = copy(P)
    P_xx_xc = reduce(lowrankdowndate, eachcol(U), init = P)
    P_post.factors[Not(consider),:] = P_xx_xc.U[Not(consider),:]
    reconstruct_consider_covariance!(P_post, P, consider)
    K, P_post
end

function reconstruct_consider_covariance!(P_post, P, consider)
    B = P_post.U[Not(consider),Not(consider)] * P_post.U[Not(consider),consider]
    P_cc = Cholesky(P.factors[consider,consider], P.uplo, 0)
    P_cc_updated = reduce(lowrankupdate, eachcol(collect(P.factors[Not(consider),consider]')), init = P_cc)
    P_cc_downdated = reduce(lowrankdowndate, eachcol(B' / P_post.L[Not(consider),Not(consider)]), init = P_cc_updated)
    P_post.factors[consider,consider] = P_cc_downdated.U
    P_post.factors[consider,Not(consider)] = P_post.factors[Not(consider),consider]'
end