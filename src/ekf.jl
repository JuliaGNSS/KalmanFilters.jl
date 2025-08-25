struct GradientOrJacobianPreparation{P<:DifferentiationInterface.Prep,F,B}
    f::F
    preparation::P
    backend::B
    contexts
end

"""
$(SIGNATURES)

JacobianPreparation calculates the Jacobian matrix automatically.
Simply pass the function that you'd like to calculate the Jacobian matrix
for and it will do so automatically. You can use different kinds of
automatic differentiator. By default it will use AutoForwardDiff.
You must also pass the state vector x in order to optimize the process.
It doesn't need to hold actual values. You can pass e.g. `zeros(num_x)`.
The type must match with the type of your state vector. With contexts,
parameters can be provided that will be passed alongside the state vector
to the function f.
"""
function JacobianPreparation(f, x::AbstractVector; backend=AutoForwardDiff(), contexts=nothing)
    GradientOrJacobianPreparation(f, prepare_jacobian(f, backend, x, Constant(contexts)), backend, contexts)
end

"""
$(SIGNATURES)

GradientPreparation calculates the gradient automatically. In contrast to 
JacobianPreparation the function `f` needs to be a scalar instead of a vector.
See JacobianPreparation for more information.
"""
function GradientPreparation(f, x::AbstractVector, backend=AutoForwardDiff(), contexts=nothing)
    GradientOrJacobianPreparation(f, prepare_gradient(f, backend, x, Constant(contexts)), backend, contexts)
end

"""
$(SIGNATURES)

GradientOrJacobianContextUpdate allows to change the context parameters.
The type of contexts must match with the context parameter provided for
JacobianPreparation.
"""
function GradientOrJacobianContextUpdate(F::GradientOrJacobianPreparation, contexts)
    GradientOrJacobianPreparation(F.f, F.preparation, F.backend, contexts)
end

"""
$(SIGNATURES)

Extended Kalman Filter time update.
F is the GradientOrJacobianPreparation object.
"""
function time_update(x, P, F::GradientOrJacobianPreparation, Q)
    F.preparation isa DifferentiationInterface.GradientPrep && error("Gradient is currently not supported for the time update.")
    x_apri, jacobian = value_and_jacobian(F.f, F.preparation, F.backend, x, F.contexts)
    P_apri = calc_apriori_covariance(P, jacobian, Q)
    KFTimeUpdate(x_apri, P_apri)
end

function value_and_gradient_or_jacobian(F::GradientOrJacobianPreparation{<:DifferentiationInterface.JacobianPrep}, x)
    value_and_jacobian(F.f, F.preparation, F.backend, x, Constant(F.contexts))
end

function value_and_gradient_or_jacobian(F::GradientOrJacobianPreparation{<:DifferentiationInterface.GradientPrep}, x)
    value, gradient = value_and_gradient(F.f, F.preparation, F.backend, x, F.contexts)
    return value, transpose(gradient)
end

"""
$(SIGNATURES)

Extended Kalman Filter measurement update.
H is the GradientOrJacobianPreparation object.
"""
function measurement_update(x, P, y, H::GradientOrJacobianPreparation, R; consider=nothing)
    y_pre, gradient_or_jacobian = value_and_gradient_or_jacobian(H, x)
    ỹ = calc_innovation(y_pre, y)
    PHᵀ = calc_P_xy(P, gradient_or_jacobian)
    S = calc_innovation_covariance(gradient_or_jacobian, P, R)
    K = calc_kalman_gain(PHᵀ, S, consider)
    x_post = calc_posterior_state(x, K, ỹ, consider)
    P_post = calc_posterior_covariance(P, PHᵀ, K, consider)
    KFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

calc_innovation(y_pre, y) = y - y_pre